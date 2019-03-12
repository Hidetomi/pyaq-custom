#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import model

from sys import stdout
from board import BSIZE, BVCNT, FEATURE_CNT
from sgf_reader import sgf2feed, import_sgf
from tqdm import trange
from parameters import PARAMS


rnd_array = [np.arange(BVCNT + 1)]
for i in range(1, 8):
    rnd_array.append(rnd_array[i - 1])
    rot_array = rnd_array[i][:BVCNT].reshape(BSIZE, BSIZE)
    if i % 2 == 0:
        rot_array = rot_array.transpose(1, 0)
    else:
        rot_array = rot_array[::-1, :]
    rnd_array[i][:BVCNT] = rot_array.reshape(BVCNT)


class Feed(object):

    def __init__(self, f_, m_, r_):
        self._feature = f_
        self._move = m_
        self._result = r_
        self.size = self._feature.shape[0]
        self._idx = 0
        self._perm = np.arange(self.size)
        np.random.shuffle(self._perm)

    def next_batch(self, batch_size=128):
        if self._idx > self.size:
            np.random.shuffle(self._perm)
            self._idx = 0
        start = self._idx
        self._idx += batch_size
        end = self._idx

        rnd_cnt = np.random.choice(np.arange(8))
        f_batch = self._feature[self._perm[start:end]]  # slice for mini-batch
        f_batch = f_batch[:, rnd_array[rnd_cnt][:BVCNT]].astype(np.float32)
        m_batch = self._move[self._perm[start:end]]  # slice for mini-batch
        m_batch = m_batch[:, rnd_array[rnd_cnt]].astype(np.float32)
        r_batch = self._result[self._perm[start:end]].astype(np.float32)

        return f_batch, m_batch, r_batch


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:
            grads.append(tf.expand_dims(g, 0))

        grad = tf.reduce_mean(tf.concat(grads, 0), 0)
        v = grad_and_vars[0][1]
        average_grads.append((grad, v))

    return average_grads


def stdout_log(str):
    stdout.write(str)
    log_file = open("learning-rate.log", "a")
    log_file.write(str)
    log_file.close()


def learn(lr_=1e-4, dr_=0.7, sgf_dir="sgf/", use_gpu=True, gpu_cnt=1):

    device_name = "gpu" if use_gpu else "cpu"
    with tf.get_default_graph().as_default(), tf.device("/cpu:0"):

        # placeholders
        f_list = []
        r_list = []
        m_list = []
        for gpu_idx in range(gpu_cnt):
            f_list.append(tf.placeholder(
                "float", shape=[None, BVCNT, FEATURE_CNT],
                name="feature_%d" % gpu_idx))
            r_list.append(tf.placeholder(
                "float", shape=[None], name="result_%d" % gpu_idx))
            m_list.append(tf.placeholder(
                "float", shape=[None, BVCNT + 1], name="move_%d" % gpu_idx))

        lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        # Adamアルゴリズム,lr=学習率
        opt = tf.train.AdamOptimizer(lr)
        dn = model.DualNetwork()

        # compute and apply gradients
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_idx in range(gpu_cnt):
                with tf.device("/%s:%d" % (device_name, gpu_idx)):

                    policy_, value_ = dn.model(
                        f_list[gpu_idx], temp=1.0, dr=dr_)
                    policy_ = tf.clip_by_value(policy_, 1e-6, 1)

                    loss_p = - \
                        tf.reduce_mean(tf.log(tf.reduce_sum(
                            tf.multiply(m_list[gpu_idx], policy_), 1)))
                    loss_v = tf.reduce_mean(
                        tf.square(tf.subtract(value_, r_list[gpu_idx])))
                    if gpu_idx == 0:
                        vars_train = tf.get_collection("vars_train")
                    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars_train])
                    loss = loss_p + 0.05 * loss_v + 1e-4 * loss_l2

                    tower_grads.append(opt.compute_gradients(loss))
                    tf.get_variable_scope().reuse_variables()

        train_op = opt.apply_gradients(average_gradients(tower_grads))

        # calculate accuracy
        with tf.variable_scope(tf.get_variable_scope(), reuse=True), \
                tf.device("/%s:0" % device_name):

            f_acc = tf.placeholder(
                "float", shape=[None, BVCNT, FEATURE_CNT], name="feature_acc")
            m_acc = tf.placeholder(
                "float", shape=[None, BVCNT + 1], name="move_acc")
            r_acc = tf.placeholder(
                "float", shape=[None], name="result_acc")

            p_, v_ = dn.model(f_acc, temp=1.0, dr=1.0)
            prediction = tf.equal(tf.reduce_max(p_, 1),
                                  tf.reduce_max(tf.multiply(p_, m_acc), 1))
            accuracy_p = tf.reduce_mean(tf.cast(prediction, "float"))
            accuracy_v = tf.reduce_mean(tf.square(tf.subtract(v_, r_acc)))
            accuracy = (accuracy_p, accuracy_v)

        sess = dn.create_sess()

    # load sgf and convert to feed
    games, all_games_total_tempo = import_sgf(sgf_dir)
    games_count = len(games)
    # stdout_log("imported %d sgf files.\n" % games_count)
    sgf_train = [games[i] for i in range(games_count) if i % 100 != 0]  # 99%
    sgf_test = [games[i] for i in range(games_count) if i % 100 == 0]  # 1%

    # stdout.write("converting ...\n")
    feed = [Feed(*(sgf2feed(sgf_train, all_games_total_tempo))),
            Feed(*(sgf2feed(sgf_test, all_games_total_tempo)))]
    feed_cnt = feed[0].size

    # learning settings
    batch_cnt = PARAMS.get('batch_cnt')
    total_epochs = PARAMS.get('total_epochs')
    epoch_steps = feed_cnt // (batch_cnt * gpu_cnt)
    # total_steps = total_epochs * epoch_steps
    global_step_idx = 0
    learning_rate = lr_

    stdout_log("learning rate=%.1g\n" % (learning_rate))
    # start_time = time.time()

    # training
    for epoch_idx in range(total_epochs):
        if epoch_idx > 0 and (epoch_idx - 8) % 8 == 0:
            learning_rate *= 0.5
            stdout_log("learning rate=%.1g\n" % (learning_rate))

        for step_idx in trange(epoch_steps, desc="Training....."):
            feed_dict_ = {}
            feed_dict_[lr] = learning_rate
            for gpu_idx in range(gpu_cnt):
                batch = feed[0].next_batch(batch_cnt)
                feed_dict_[f_list[gpu_idx]] = np.array(batch[0])
                feed_dict_[m_list[gpu_idx]] = np.array(batch[1])
                feed_dict_[r_list[gpu_idx]] = np.array(batch[2])

            sess.run(train_op, feed_dict=feed_dict_)
            global_step_idx += 1

        acc_steps = feed[1].size // batch_cnt
        np.random.shuffle(feed[0]._perm)

        for _ in range(acc_steps):
            acc_batch = feed[i].next_batch(batch_cnt)
            sess.run(accuracy, feed_dict={f_acc: acc_batch[0],
                                          m_acc: acc_batch[1],
                                          r_acc: acc_batch[2]})

    dn.save_vars(sess, "model.ckpt")
