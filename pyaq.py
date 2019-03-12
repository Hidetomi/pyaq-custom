#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import sys
from board import BVCNT, PASS, Board, rv2ev
import gtp
import learn
import search
import numpy as np
from parameter_loader import ParameterLoader
from parameters import PARAMS

if __name__ == "__main__":

    ParameterLoader()
    launch_mode = PARAMS.get("launch_mode")
    model = PARAMS.get("model")
    sgf_dir = PARAMS.get("sgf_dir")
    use_gpu = PARAMS.get("use_gpu")
    main_time = PARAMS.get("main_time")
    byoyomi = PARAMS.get("byoyomi")
    quick = PARAMS.get("quick")
    random = PARAMS.get("random")
    clean = PARAMS.get("clean")

    if launch_mode == 0:
        gtp.call_gtp(main_time, byoyomi, quick, clean, use_gpu)

    elif launch_mode == 1:
        b = Board()
        if not random:
            tree = search.Tree(model, use_gpu)

        while b.move_cnt < BVCNT * 2:
            prev_move = b.prev_move
            if random:
                move = b.random_play()
            elif quick:
                move = rv2ev(np.argmax(tree.evaluate(b)[0][0]))
                b.play(move, False)
            else:
                move, _ = tree.search(b, 0, clean=clean)
                b.play(move, False)

            b.showboard()
            if prev_move == PASS and move == PASS:
                break

        score_list = []
        b_cpy = Board()

        for i in range(256):
            b.copy(b_cpy)
            b_cpy.rollout(show_board=False)
            score_list.append(b_cpy.score())

        score = Counter(score_list).most_common(1)[0][0]
        if score == 0:
            result_str = "Draw"
        else:
            winner = "B" if score > 0 else "W"
            result_str = "%s+%.1f" % (winner, abs(score))
        sys.stderr.write("result: %s\n" % result_str)

    else:
        learn.learn(3e-4, 0.5, sgf_dir=sgf_dir, use_gpu=use_gpu, gpu_cnt=1)
