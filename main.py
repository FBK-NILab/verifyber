#!/usr/bin/env python

import os
import sys
import argparse
import configparser
import glob

from loops.test import test
from loops.train import train
from utils.general_utils import get_cfg_value, print_cfg, set_seed

if __name__ == '__main__':


    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('var', nargs='?', const=0, default='DEFAULT',
                        help='The tag for the configuration file.')
    parser.add_argument('-opt', nargs='?', const=0, default='train',
                        help='type of exec: train | test')
    parser.add_argument('--exp', nargs='?', const=0, default='',
                        help='experiment path')
    parser.add_argument('--weights', nargs='?', const=0, default='',
                        help='file of the saved model')
    parser.add_argument('--sub_list', nargs='?', const=0, default='',
                        help='sub list containing the test subjects')
    parser.add_argument('--root_dir', nargs='?', const=0, default='',
                        help='dataset root dir')
    parser.add_argument('--config', nargs='?', const=1, default='',
                        help='load config.txt in exp dir')
    parser.add_argument('--with_gt', nargs='?', const=1, default='',
                        help='if gt is available')
    parser.add_argument('--save_pred', nargs='?', const=1, default='',
                        help='if present save prediction no otherwise')
    args = parser.parse_args()


    #### CONFIG PARSING

    # Reading configuration file with specific setting for a run.
    # Mandatory variables for this script:

    cfg_parser = configparser.ConfigParser(inline_comment_prefixes="#")
    if not args.config and args.exp:
        cfg_parser.read(args.exp + '/config.txt')
    elif args.config:
        cfg_parser.read(args.config)
    else:
        cfg_parser.read(glob.glob('configs/*.cfg'))
    cfg = {}
    cfg[args.var] = {}
    if 'DEFAULT' in cfg_parser:
        for name, value in cfg_parser.items('DEFAULT'):
            cfg[args.var][name] = get_cfg_value(value)
    for name, value in cfg_parser.items(args.var):
        cfg[args.var][name] = get_cfg_value(value)
    for k in cfg_parser.sections():
        if k != args.var:
            del cfg_parser[k]
    cfg[args.var]['cfg_parser_obj'] = cfg_parser
    
    cfg['opt'] = args.opt

    set_seed(cfg[args.var]['seed'])

    #### LAUNCH RUNS
    if cfg['opt'] == 'train':
        print_cfg(cfg[args.var])
        train(cfg[args.var])
    if cfg['opt'] == 'test':
        if not args.exp:
            sys.exit('Missing argument --exp')
        cfg[args.var]['exp_path'] = args.exp
        if args.weights:
            cfg[args.var]['weights_path'] = args.weights
        else:
            cfg[args.var]['weights_path'] = ''

        if args.sub_list:
            cfg[args.var]['sub_list_test'] = args.sub_list

        if args.root_dir:
            cfg[args.var]['val_dataset_dir'] = args.root_dir

        if args.with_gt:
            cfg[args.var]['with_gt'] = True
        else:
            cfg[args.var]['with_gt'] = False

        if args.save_pred:
            cfg[args.var]['save_pred'] = True
        else:
            cfg[args.var]['save_pred'] = False

        print_cfg(cfg[args.var])
        test(cfg[args.var])

