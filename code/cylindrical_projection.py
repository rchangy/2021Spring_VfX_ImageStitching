import numpy as np
import cv2 as cv
import sys
import math
import csv
import os
import cylindrical_projection

def cylindricalProjection(I, S, img, f):
    # inverse
    # x = f * tan(x'/f)
    # y = y' * (x**2 + f**2)**0.5 / f
    # x' = f * tan-1(x/f)
    # y' = f * (y / (x**2 + f**2))
    m, n, chn = img.shape
    x_coord, y_coord = np.meshgrid(range(n), range(m))
    x_coord -= int(n/2)
    y_coord = -y_coord + int(m/2)
    x_ori = np.round(f * np.tan(x_coord / f)).astype(int)
    y_ori = np.round(y_coord * (x_ori**2 + f**2)**0.5 / f).astype(int)

    x_ij = -(y_ori - int(m/2)).flatten()
    y_ij = (x_ori + int(n/2)).flatten()
    x_ij_max = np.where(x_ij >= m)
    x_ij_min = np.where(x_ij < 0)
    y_ij_max = np.where(y_ij >= n)
    y_ij_min = np.where(y_ij < 0)
    x_ij[x_ij_max] = 0
    x_ij[x_ij_min] = 0
    y_ij[y_ij_max] = 0
    y_ij[y_ij_min] = 0

    img_t = img[tuple([x_ij, y_ij])][:]
    img_t[x_ij_max] = np.zeros(3)
    img_t[x_ij_min] = np.zeros(3)
    img_t[y_ij_max] = np.zeros(3)
    img_t[y_ij_min] = np.zeros(3)
    img_t = img_t.reshape((m, n, 3))
    I_t = []
    for i in range(S):
        I_x = I[i][1] - int(n/2)
        I_y = -I[i][0] + int(m/2)
        I_xt = f * np.arctan(I_x/f)
        I_yt = f * (I_y / (I_x**2 + f**2))
        I_xt_ij = -(I_yt - int(m/2))
        I_yt_ij = I_xt + int(n/2)
        I_t.append(tuple([I_xt_ij, I_yt_ij]))
    return img_t, I_t
