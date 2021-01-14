from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme

import json
import shutil as sh
from .labelmeutils import Labelme2Vocor

def main_labelme2voc(args, ordered_keys=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args(args)

    vocor = Labelme2Vocor(args.input_dir, args.output_dir, args.labels, noviz=args.noviz, debug=False)
    vocor.getClasses()
    vocor.output(ordered_keys=ordered_keys)


from .labelmeutils import Labelme2Cocoer

def getParser_labelme2coco():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--debug", action="store_true", default=False, help="labels file")
    return parser


def main_labelme2coco(arguments=None):
    parser = getParser_labelme2coco()
    if arguments is None:
        arguments = sys.argv[1:]
    args = parser.parse_args(arguments)

    if osp.exists(args.output_dir):
        print("Output directory already exists: ", args.output_dir)
        sh.rmtree(args.output_dir)
        # sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    print("Creating dataset:", args.output_dir)

    label_files = glob.glob(osp.join(args.input_dir, "*.json"))

    cocoGen = Labelme2Cocoer(args.output_dir, debug=args.debug)
    cocoGen.classNameToId(args.labels)
    cocoGen.generateCocoJson(label_files)
    cocoGen.output()
    return cocoGen