import numpy as np
import sys
import math
import csv
from scipy.spatial import KDTree
import random
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
import blending

def readinH(directory):
    with open(f'{directory}/H.csv', 'r') as H_in:
        rows = csv.reader(H_in)
        # H = [np.array(row).astype(float).reshape((3, 3)) for row in rows]
        H = [np.array(row).astype(int) for row in rows]
    H = np.array(H)
    return H

def readImages(directory):
    with open(f'{directory}/imlist.txt', 'r') as in_f:
        P = int(in_f.readline())
    imgs = [cv.imread(f'{directory}/projected_img/{i}.png') for i in range(P)]
    return imgs, P

def Blending(full_image, full_image_label, img, H, width):
    m, n, ch = img.shape
    paste_area = full_image_label[H[0]:H[0]+m, H[1]:H[1]+n]
    patch = full_image[H[0]:H[0]+m, H[1]:H[1]+n]
    blended, paste_area = blending.multiBand(patch, img, paste_area, 40)
    full_image[H[0]:H[0]+m, H[1]:H[1]+n] = blended
    full_image_label[H[0]:H[0]+m, H[1]:H[1]+n] = paste_area
    return full_image


def Stitching(P, imgs, H):
    # Path(f"{directory}/steps").mkdir(parents=True, exist_ok=True)
    H_cum = np.cumsum(H, axis = 0)
    H_adjust = H_cum.T
    H_adjust[0] -= H_adjust[0].min()
    H_adjust[1] -= H_adjust[1].min()
    full_image = np.zeros((H_adjust[0].max() + imgs[H_adjust[0].argmin()].shape[0], H_adjust[1].max() + imgs[H_adjust[1].argmin()].shape[1], 3))
    full_image_label = np.zeros((full_image.shape[0], full_image.shape[1]), dtype=bool)
    H_adjust = H_adjust.T
    for i in range(P):
        full_image = Blending(full_image, full_image_label,imgs[i], H_adjust[i], 20)
        # cv.imwrite(f'{directory}/steps/{i}.png', full_image)
        print(f'image {i} blended')
    return full_image

directory = sys.argv[1]
imgs , P = readImages(directory)
H = readinH(directory)
full_image = Stitching(P, imgs, H)
cv.imwrite(f'{directory}/full_image.png', full_image)