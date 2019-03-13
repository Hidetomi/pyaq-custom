#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser as cp
import os
import sys


class ParameterLoader(object):

    def __init__(self):
        self.__ARGS = sys.argv
        self.__PATH = os.getcwd() + "/"
        self.__CONFIG_FILE = None
        self.__params = {}
        self.__deploy()

    def __deploy(self):
        self.__analyzeArgs()
        self.__readConfigFile()

    def __analyzeArgs(self):
        self.__params['launch_mode'] = 0  # 0: gtp, 1: self, 2: learn
        self.__params['quick'] = False
        self.__params['random'] = False
        self.__params['clean'] = False
        self.__params['use_gpu'] = True

        for arg in self.__ARGS:
            if arg.find("self") >= 0:
                self.__params['launch_mode'] = 1
            elif arg.find("learn") >= 0:
                self.__params['launch_mode'] = 2
            elif arg.find("quick") >= 0:
                self.__params['quick'] = True
            elif arg.find("random") >= 0:
                self.__params['random'] = True
            elif arg.find("clean") >= 0:
                self.__params['clean'] = True
            elif arg.find("config_file") >= 0:
                self.__params['config_file'] = arg[arg.find("=") + 1:]

    def __readConfigFile(self):
        self.__CONFIG_FILE = self.__PATH + self.__params['config_file']
        if not os.path.exists(self.__CONFIG_FILE):
            self.__CONFIG_FILE = "example.ini"

        config = cp.SafeConfigParser()
        config.read(self.__CONFIG_FILE)

        # DEFUALT
        self.__params['sgf_dir'] = config['DEFAULT'].get('sgf_dir')

        # TENSORFLOW'S
        self.__params['use_gpu'] = config['TENSORFLOW'].getboolean('use_gpu')
        self.__params['model'] = config['TENSORFLOW'].get('model')

        # LEARNING
        self.__params['batch_cnt'] = config['LEARNING'].getint('batch_cnt')
        self.__params['total_epochs'] = config['LEARNING'].getint(
            'total_epochs')
        self.__params['keep_previous_count'] = config['LEARNING'].getint(
            'keep_previous_count')
        self.__params['filter_count'] = config['LEARNING'].getint(
            'filter_count')
        self.__params['block_count'] = config['LEARNING'].getint('block_count')
        self.__params['w_wdt'] = config['LEARNING'].getfloat('w_wdt')
        self.__params['b_wdt'] = config['LEARNING'].getfloat('b_wdt')

        # GO
        self.__params['board_size'] = config['GO'].getint('board_size')
        self.__params['komi'] = config['GO'].getfloat('komi')
        self.__params['main_time'] = config['GO'].getfloat('main_time')
        self.__params['byoyomi'] = config['GO'].getfloat('byoyomi')

    def get(self, name):
        return self.__params.get(name)
