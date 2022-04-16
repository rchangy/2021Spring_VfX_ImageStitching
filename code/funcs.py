import numpy as np
import cv2 as cv
import math
from scipy.interpolate import griddata
from scipy.signal import convolve2d

def getGaussianMask(sigma, half_size):
    S = 2*sigma**2
    size = half_size*2 + 1
    kernel = np.zeros((size, size))
    for x  in range(-half_size, half_size+1):
        for y in range(-half_size, half_size+1):
            kernel[x + half_size, y + half_size] = -(x**2 + y**2) / S
    kernel = np.exp(kernel)
    return kernel

def Gaussian(img, sigma, half_size, gray=True):
    kernel = getGaussianMask(sigma, half_size)
    kernel /= kernel.sum()
    if gray:
        filtered_img = convolve2d(img, kernel, mode='same', boundary='symm')
    else:
        m, n, ch = img.shape
        filtered_img = np.zeros_like(img)
        for i in range(ch):
            filtered_img[:, :, i] = (convolve2d(img[:, :, i], kernel, mode='same', boundary='symm'))
    return filtered_img 

def Sampling(img, rate = 2, sigma=1, gray=True):
    if gray:
        m, n = img.shape
        m_s = round(m/rate)
        n_s = round(n/rate)
        img_g = Gaussian(img, sigma, int(np.ceil(sigma)))
        img_sampling = np.zeros((m_s, n_s))
        count = np.zeros((m_s, n_s))
        for i in range(m):
            for j in range(n):
                img_sampling[min(round(i/rate), m_s-1), min(round(j/rate), n_s-1)] += img[i, j]
                count[min(round(i/rate), m_s-1), min(round(j/rate), n_s-1)] += 1
    else:
        m, n, ch = img.shape
        
        m_s = round(m/rate)
        n_s = round(n/rate)
        # print(m, n, m_s, n_s)
        img_g = Gaussian(img, sigma, int(np.ceil(sigma)), gray)
        img_sampling = np.zeros((m_s, n_s, ch))
        count = np.zeros((m_s, n_s, ch))
        for i in range(m):
            for j in range(n):
                img_sampling[min(round(i/rate), m_s-1), min(round(j/rate), n_s-1), :] += img[i, j, :]
                count[min(round(i/rate), m_s-1), min(round(j/rate), n_s-1), :] += 1
    img_sampling  = img_sampling / count
    return img_sampling

def multiScaling(img, levels, gray=True):
    multi_scaled_img = []
    multi_scaled_img.append(img)
    for l in range(levels-1):
        img_new = Sampling(multi_scaled_img[-1], gray=gray)
        multi_scaled_img.append(img_new)
    return multi_scaled_img

def Upsampling(img, shape, gray=True):
    if gray:
        m, n = shape
    else:
        m_ori, n_ori, ch = img.shape
        m, n, ch = shape
        img_ret = np.zeros((m*n, ch))
        img_index = np.meshgrid(range(m_ori), range(n_ori), indexing='ij')
        img_index = np.array([img_index[0].flatten(), img_index[1].flatten()]).T
        ret_index = np.meshgrid(range(m), range(n), indexing='ij')
        ret_index = np.array([(ret_index[0] * ((m_ori-1) / (m-1))).flatten(), (ret_index[1] * ((n_ori-1) / (n-1))).flatten()]).T
        for i in range(ch):
            img_ret[:, i] = griddata(img_index, img[:, :, i].flatten(), ret_index)
        img_ret = img_ret.reshape(shape)
    return img_ret

def partialDerivative(img, sigma=1, gray=True):
    img = img.astype(int)
    if gray:
        img_g = Gaussian(img, sigma, round(sigma))
        kernel_x = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
        kernel_y = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2
        img_dx = convolve2d(img_g, np.flip(kernel_x), mode='same', boundary='symm')
        img_dy = convolve2d(img_g, np.flip(kernel_y), mode='same', boundary='symm')
        return img_dx, img_dy
    else:
        if sigma != 0:
            img_g = Gaussian(img, sigma, round(sigma), gray=False)
        else:
            img_g = img
        kernel_x = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
        kernel_y = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2
        m, n, ch = img_g.shape
        img_dx, img_dy = np.zeros_like(img_g), np.zeros_like(img_g)
        for i in range(ch):
            img_dx[:, :, i] = convolve2d(img_g[:, :, i], np.flip(kernel_x), mode='same', boundary='symm')
            img_dy[:, :, i] = convolve2d(img_g[:, :, i], np.flip(kernel_y), mode='same', boundary='symm')
        return img_dx, img_dy