#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser as cp
import os
import sys
from parameters import PARAMS


class ParameterLoader(object):

    def __init__(self):
        self.__ARGS = sys.argv
        self.__PATH = os.getcwd() + "/"
        self.__CONFIG_FILE = None
        self.__deploy()

    def __deploy(self):
        self.__analyzeArgs()
        self.__readConfigFile()

    def __analyzeArgs(self):
        PARAMS['launch_mode'] = 0  # 0: gtp, 1: self, 2: learn
        PARAMS['byoyomi'] = 3.0
        PARAMS['main_time'] = 0.0
        PARAMS['quick'] = False
        PARAMS['random'] = False
        PARAMS['clean'] = False
        PARAMS['use_gpu'] = True

        for arg in self.__ARGS:
            if arg.find("self") >= 0:
                PARAMS['launch_mode'] = 1
            elif arg.find("learn") >= 0:
                PARAMS['launch_mode'] = 2
            elif arg.find("quick") >= 0:
                PARAMS['quick'] = True
            elif arg.find("random") >= 0:
                PARAMS['random'] = True
            elif arg.find("clean") >= 0:
                PARAMS['clean'] = True
            elif arg.find("main_time") >= 0:
                PARAMS['main_time'] = float(arg[arg.find("=") + 1:])
            elif arg.find("byoyomi") >= 0:
                PARAMS['byoyomi'] = float(arg[arg.find("=") + 1:])
            elif arg.find("cpu") >= 0:
                PARAMS['use_gpu'] = False
            elif arg.find("config_file") >= 0:
                PARAMS['config_file'] = arg[arg.find("=") + 1:]

    def __readConfigFile(self):
        self.__CONFIG_FILE = self.__PATH + PARAMS['config_file']
        if not os.path.exists(self.__CONFIG_FILE):
            sys.stderr.write('%s が見つかりません' % self.__CONFIG_FILE)
            sys.exit(2)

        config = cp.SafeConfigParser()
        config.read(self.__CONFIG_FILE)

        # DEFUALT
        PARAMS['sgf_dir'] = config['DEFAULT'].get('sgf_dir')

        # TENSORFLOW'S
        PARAMS['use_gpu'] = config['TENSORFLOW'].getboolean('use_gpu')
        PARAMS['model'] = config['TENSORFLOW'].get('model')

        # LEARNING
        PARAMS['batch_cnt'] = config['LEARNING'].getint('batch_cnt')
        PARAMS['total_epochs'] = config['LEARNING'].getint('total_epochs')

        # GO
