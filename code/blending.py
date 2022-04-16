import numpy as np
import cv2 as cv
import sys
import math
from scipy.interpolate import griddata
from scipy.signal import convolve2d
import csv
import os
import funcs


def preprocessing(full, img, width, ori_shape, pos):
    max_x, min_x, max_y, min_y = pos
    m, n, ch = ori_shape
    w_m = max_x - min_x + 1
    w_n = max_y - min_y + 1
    w = np.zeros((w_m, w_n))
    sub_full = np.zeros((w_m, w_n, ch))
    sub_img = np.zeros((w_m, w_n, ch))
    center_column = w_n / 2
    half_width = width / 2

    if max_y == n-1:
        w[:, :int(center_column)] = 1
        sub_img[:, :int(center_column + half_width)] = img[:, :int(center_column + half_width)]
        sub_img[:, :int(center_column + half_width)][sub_img[:, :int(center_column + half_width)].sum(axis=2) == 0] =  full[:, :int(center_column + half_width)][sub_img[:, :int(center_column + half_width)].sum(axis=2) == 0]
        sub_full[:, int(center_column - half_width):] = full[:, int(center_column - half_width):]
        sub_full[:, int(center_column - half_width):][sub_full[:, int(center_column - half_width):].sum(axis==2) == 0] = img[:, int(center_column - half_width):][sub_full[:, int(center_column - half_width):].sum(axis==2) == 0]
    elif min_y == 0:
        w[:, int(center_column): ] = 1
        sub_full[:, :int(center_column + half_width)] = full[:, :int(center_column + half_width)]
        sub_full[:, :int(center_column + half_width)][sub_full[:, :int(center_column + half_width)].sum(axis=2) == 0] = img[:, :int(center_column + half_width)][sub_full[:, :int(center_column + half_width)].sum(axis=2) == 0] 
        sub_img[:, int(center_column - half_width):] = img[:, int(center_column - half_width):]
        sub_img[:, int(center_column - half_width):][sub_img[:, int(center_column - half_width):].sum(axis=2)==0] = full[:, int(center_column - half_width):][sub_img[:, int(center_column - half_width):].sum(axis=2)==0] 
    return sub_full, sub_img, w

def LaplacianPyramid(img, levels):
    gaussian_pyramid = funcs.multiScaling(img, levels, gray=False)
    laplacian_pyramid = []
    for i in range(levels-1):
        g_img = funcs.Gaussian(gaussian_pyramid[i], 1, 1, gray=False)
        laplacian_pyramid.append(gaussian_pyramid[i] - g_img)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def Collapse(blended):
    img_ret = blended[-1]
    for img in blended[-2::-1]:
        img_ret = funcs.Upsampling(img_ret, img.shape, gray = False)
        img_ret += img
    return img_ret

def multiBand(full_img_patch, img, labels, width):
    #preprocessing
    m, n, ch = img.shape
    I = np.where(labels == True)
    if I[0].size == 0:
        label_ret = np.where(img.sum(axis=2) != 0, True, False)
        return img, label_ret
    max_x, min_x = I[0].max(), I[0].min()
    max_y, min_y = I[1].max(), I[1].min()
    pos = tuple([max_x, min_x, max_y, min_y])
    overlapped_full = full_img_patch[min_x:max_x+1, min_y:max_y+1]
    overlapped_img = img[min_x:max_x+1, min_y:max_y+1]
    sub_full, sub_img, w = preprocessing(overlapped_full, overlapped_img, width, img.shape, pos)
    #construct pyramids
    max_l = 3
    gaussian_py = funcs.multiScaling(w, max_l)
    img_py = LaplacianPyramid(sub_img, max_l)
    full_py = LaplacianPyramid(sub_full, max_l)

    #blending
    blended = []
    for i in range(max_l):
        w_m, w_n = gaussian_py[i].shape
        weight = np.repeat(gaussian_py[i].reshape(w_m, w_n, 1), 3, axis=2)
        blended.append(img_py[i] * weight + full_py[i] * (1 - weight))
    
    #collapse pyramid
    img_blended = Collapse(blended)
    img_blended[img_blended > 255] = 255
    img_blended[img_blended < 0] = 0
    img_ret = img.copy()
    img_ret[min_x:max_x+1, min_y:max_y+1] = img_blended
    label_ret = np.where(img_ret.sum(axis=2) != 0, True, False)
    return img_ret, label_ret

def forwardDifference(img):
    img_g = img.astype(int)
    kernel_x = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]) / 2
    kernel_y = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]) / 2
    m, n, ch = img_g.shape
    img_dx, img_dy = np.zeros_like(img_g), np.zeros_like(img_g)
    for i in range(ch):
        img_dx[:, :, i] = convolve2d(img_g[:, :, i], np.flip(kernel_x), mode='same', boundary='symm')
        img_dy[:, :, i] = convolve2d(img_g[:, :, i], np.flip(kernel_y), mode='same', boundary='symm')
    return img_dx, img_dy
