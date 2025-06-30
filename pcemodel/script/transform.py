import os, shutil
import numpy as np
import cv2 as cv
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output', type=str, default=True)
    args = parser.parse_args()
    return args


def process_path(path:str, output:str):
    spath = os.path.join(path, "val", "images")

    for fi in os.listdir(spath):
        fpath = os.path.join(spath, fi)
        di = fi.split("_")[0]
        dpath = os.path.join(output, di)
        os.makedirs(dpath, exist_ok=True)
        
        if int(di) % 2 == 0:
            shutil.move(fpath, os.path.join(dpath, fi))
        else:
            img = Image.open(fpath)
            # apply flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(os.path.join(dpath, fi))

if __name__ == '__main__':

    args = parse_args()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    for s in range(args.split):
        process_path(args.path+f'_{s}', args.output)
