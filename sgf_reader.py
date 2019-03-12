#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
from board import BSIZE, BVCNT, Board, ev2rv
import numpy as np
from tqdm import trange

import sgf


def import_sgf(dir_path):
    all_games_total_tempo = 0
    dir_path += "/*.sgf"
    file_list = glob.glob(dir_path)
    # sd_list = []
    games = []
    # b = Board()

    for i in trange(len(file_list), desc='Reading......'):
        # f = file_list[i]
        # sd_list.append(sgf_data())
        # sd_list[-1].import_file(f)

        with open(file_list[i], "r") as f:
            try:
                for game in sgf.parse(f.read()):
                    games.append(game)
                    all_games_total_tempo += len(game.nodes)-1
            except Exception as e:
                print(e)
                continue

    return games, all_games_total_tempo


def sgf2feed(games, all_games_total_tempo):
    # 配列数が全局全手数の総和の、全交点数（81）×7である、8ビットの符号なし整数の三次元配列
    feature = np.zeros((all_games_total_tempo, BVCNT, 7), dtype=np.uint8)
    # 全局全手数の総和×全交点数+1である、8ビットの符号なし整数の二次元配列
    move = np.zeros((all_games_total_tempo, BVCNT + 1), dtype=np.uint8)
    # 全局全手数の総和である、8ビットの符号なし整数の二次元配列
    result = np.zeros((all_games_total_tempo), dtype=np.int8)

    b = Board()
    re = 0
    for i in trange(len(games), desc="Converting..."):
        game = games[i]
        # 対局情報取得
        game_information = game.nodes[0].properties
        board_size = int(game_information.get('SZ')[0])
        handy_cap = 0 \
            if game_information.get('HA') is None \
            else int(game_information.get('HA'))

        # 指定した碁盤のサイズ以外、置碁は対象外
        if board_size != BSIZE or handy_cap != 0:
            continue

        # 対局結果取得
        re = re2dec(game_information.get('RE')[0])
        if re == 0:
            continue

        # 一手目から開始
        for j, node in enumerate(game):
            if node.first:
                continue

            train_idx = j - 1
            b.clear()
            feature[train_idx] = b.feature()
            move[train_idx, ev2rv(sgf2ev(node.current_prop_value[0]))] = 1
            result[train_idx] = re * (2 * b.turn - 1)
            b.play(sgf2ev(node.current_prop_value[0]), False)

    return feature, move, result


def re2dec(result):
    if result == '0':
        return 0
    _re = result.split("+")
    if _re[0] == 'B':
        return -1 if _re[1] == 'R' else 1
    else:
        return 1 if _re[1] == 'R' else -1


def sgf2ev(v_sgf):
    # TODO この関数は暫定
    if len(v_sgf) != 2:
        # 9路盤の外周2目広げた全交点(121)
        return (BSIZE + 2) ** 2
    # 盤の交点英字(19路)
    labels = "abcdefghijklmnopqrs"
    x = labels.find(v_sgf[0]) + 1
    y = labels.find(v_sgf[1]) + 1
    return x + (BSIZE + 1 - y) * (BSIZE + 2)
