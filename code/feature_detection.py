import numpy as np
import cv2 as cv
import sys
import math
from scipy.interpolate import griddata
from scipy.signal import convolve2d
import csv
import os
import cylindrical_projection as cproj
from pathlib import Path
import funcs

def RGBtoGray(imgs, P):
    w = [0.2989, 0.5870, 0.1140]
    m, n, chn = imgs[0].shape
    imgs_gray = list()
    img_gray = np.zeros((m, n))
    for i in range(P):
        img_gray = np.matmul(imgs[i], w)
        imgs_gray.append(img_gray)
    return imgs_gray

def readImages(directory):
    with open(f'{directory}/imlist.txt', 'r') as in_f:
        P = int(in_f.readline())
        img_path = [in_f.readline().strip() for i in range(P)]
        imgs = [cv.imread(f'{directory}/{img_path[i]}') for i in range(P)]
    return imgs, P

def cornerResponse(img):
    img_dx, img_dy = funcs.partialDerivative(img)
    I_x2 = img_dx **2
    I_y2 = img_dy **2
    I_xy = img_dy * img_dx
    S_x2 = funcs.Gaussian(I_x2, 1.5, 2)
    S_y2 = funcs.Gaussian(I_y2, 1.5, 2)
    S_xy = funcs.Gaussian(I_xy, 1.5, 2)
    det = S_x2 * S_y2 - S_xy * S_xy
    trace = S_x2 + S_y2
    trace_cor = np.where(trace == 0, -1, trace)
    R = det / trace_cor
    R = np.where(trace == 0, 0, R)
    return R

def cornerResponseThresholding(R):
    I = np.where(R >= 10)
    remove = list()
    m, n = R.shape
    for i in range(I[0].size):
        local = R[max(I[0][i]-1, 0):min(I[0][i]+2, m), max(I[1][i]-1, 0):min(I[1][i]+2, n)]
        if local.max() != R[I[0][i], I[1][i]]:
            remove.append(i)
    I = tuple([np.delete(I[0], remove), np.delete(I[1], remove)])
    return I

def nonMaximalSuppression(I, R, num, offsets):
    min_supression_radius = np.array([-1.] * I[0].size)
    for i in range(I[0].size):
        J = np.where(R * 0.9 > R[I[0][i], I[1][i]])
        if J[0].size > 0:
            dist = (I[0][i] - J[0])**2 + (I[1][i] - J[1])**2
            min_supression_radius[i] = dist.min()
        else:
            min_supression_radius[i] = math.inf
    sorted_arg = np.argsort(-min_supression_radius)
    num = min(num, I[0].size)
    I = tuple([I[0][sorted_arg[0:num]], I[1][sorted_arg[0:num]]])
    offsets = tuple([offsets[0][sorted_arg[0:num]], offsets[1][sorted_arg[0:num]]])
    return I, offsets

def localDerivatives(img):
    kernel_x = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    kernel_y = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2
    d_x = convolve2d(img, kernel_x, mode='same', boundary='symm')
    d_y = convolve2d(img, kernel_y, mode='same', boundary='symm')
    kernel_x2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    kernel_y2 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    d_x2 = convolve2d(img, kernel_x2, mode='same', boundary='symm')
    d_y2 = convolve2d(img, kernel_y2, mode='same', boundary='symm')
    kernel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])/4
    d_xy = convolve2d(img, kernel_xy, mode='same', boundary='symm')
    return d_x, d_y, d_x2, d_y2, d_xy

def subPixelRefinement(I, R):
    d_x, d_y, d_x2, d_y2, d_xy = localDerivatives(R)
    m, n = R.shape
    d_matrix = np.zeros((m, n, 2, 2))
    d_matrix[:, :, 0, 0] = d_x2
    d_matrix[:, :, 0, 1] = d_xy
    d_matrix[:, :, 1, 0] = d_xy
    d_matrix[:, :, 1, 1] = d_y2
    offsets = np.zeros((2, I[0].size))
    remove = list()
    for i in range(I[0].size):
        M = d_matrix[I[0][i], I[1][i]]
        M_inv = np.linalg.inv(M)
        offset = np.matmul(-M_inv, np.array([d_x[I[0][i], I[1][i]], d_y[I[0][i], I[1][i]]]).reshape(2, 1))
        if abs(offset[0]) >= 1.5 or abs(offset[1]) >= 1.5:
            remove.append(i)
        else:
            D_x = (d_x[I[0][i], I[1][i]] * offset[0] + d_y[I[0][i], I[1][i]] * offset[1]) + 0.5 * np.matmul(offset.T, np.matmul(M, np.array([offset[0], offset[1]]).reshape(2, 1)))
            if D_x >= 0:
                R[I[0][i], I[1][i]] += D_x
                I[0][i] = min(max(I[0][i]+offset[0], 0), m-1)
                I[1][i] = min(max(I[1][i]+offset[1], 0), n-1)
            else:
                offset[0], offset[1] = 0, 0
        offsets[0][i] = offset[0]
        offsets[1][i] = offset[1]
    I = tuple([np.delete(I[0], remove), np.delete(I[1], remove)])
    offsets = tuple([np.delete(offsets[0], remove), np.delete(offsets[1], remove)])

    check = np.where(abs(offsets[0]) >= 1.5)
    check_2 = np.where(abs(offsets[1]) >= 1.5)
    return I, offsets

def orientationAssignment(I, corner_response):
    d_x, d_y = funcs.partialDerivative(corner_response, sigma=4.5)
    orientation = tuple([d_x[I].flatten(), d_y[I].flatten()])
    length = (orientation[0]**2 + orientation[1]**2)**0.5
    orientation = tuple([orientation[0] / length, orientation[1] / length])
    return orientation

def Padding(img, bound, type):
    m, n = img.shape
    m_pad = m+bound*2
    n_pad = n+bound*2
    if type=='even':
        p = 0
    elif type=='odd':
        p = 1
    img_pad = np.zeros((m_pad, n_pad), dtype=np.uint8)
    img_pad[bound:m_pad-bound, bound:n_pad-bound] = img
    img_pad[bound:m_pad-bound, 0:bound] = np.fliplr(img[:, p:bound+p])
    img_pad[bound:m_pad-bound, n_pad-bound:n_pad] = np.fliplr(img[:, n-p-bound:n-p])
    img_pad[0:bound, bound:n_pad-bound] = np.flipud(img[p:bound+p, :])
    img_pad[m_pad-bound:m_pad, bound:n_pad-bound] = np.flipud(img[m-p-bound:m-p, :])
    img_pad[0:bound, 0:bound] = np.flip(img[p:bound+p, p:bound+p])
    img_pad[m_pad-bound:m_pad, 0:bound] = np.flip(img[m-p-bound:m-p, p:bound+p] )
    img_pad[0:bound, n_pad-bound:n_pad] = np.flip(img[p:bound+p, n-p-bound:n-p])
    img_pad[m_pad-bound:m_pad, n_pad-bound:n_pad] = np.flip(img[m-p-bound:m-p, n-p-bound:n-p])
    return img_pad

def getDiscriptorVector(I, O, offset, img):
    descriptors = np.zeros((I[0].size, 8, 8))
    img_padded = Padding(img, 30, 'even')
    m, n = img_padded.shape
    pos = np.meshgrid(range(40), range(40), indexing='ij')
    pos = np.array([pos[0].flatten()-19.5, pos[1].flatten()-19.5])
    for i in range(I[0].size):
        d_x, d_y = O[0][i], O[1][i]
        pos_rotate = np.matmul(np.array([[d_y, -d_x],[d_x, d_y]]), pos)
        pos_rotate[0] += I[0][i] + 30 + offset[0][i]
        pos_rotate[1] += I[1][i] + 30 + offset[1][i]

        # min_x, max_x = math.floor(pos_rotate[0].min()), math.ceil(pos_rotate[0].max())+1
        # min_y, max_y = math.floor(pos_rotate[1].min()), math.ceil(pos_rotate[1].max())+1
        # points = np.meshgrid(range(min_x, max_x), range(min_y, max_y), indexing='ij')
        # points = np.array([points[0].flatten(), points[1].flatten()])
        # values = griddata(points.T, img_padded[min_x:max_x, min_y:max_y].flatten(), pos_rotate.T)
        # print(pos_rotate)
        values = img_padded[tuple(pos_rotate.astype(int))]

        values = values.reshape((40, 40))
        descriptors[i] = funcs.Sampling(values, rate = 5, sigma=1.5)
        descriptors[i] = (descriptors[i] - descriptors[i].mean()) / descriptors[i].std()
    return descriptors

def writeDiscriptors(P, S, D, I, discriptor_file, index_file, shape):
    with open(index_file, 'w') as index_f:
        index_writer = csv.writer(index_f)
        index_writer.writerow([P, S])
        with open(discriptor_file, 'w') as dis_f:
            dis_writer = csv.writer(dis_f)
            # write different scales first
            for i in range(P):
                for j in range(S):
                    D[i][j] = D[i][j].reshape((-1, 64))
                    dis_writer.writerows(D[i][j])
                    I[i][j] = np.array(I[i][j])
                    if I[i][j][0].size == 0:
                        I[i][j] = np.array([[-1], [-1]])
                    if i > 0:
                        up_sampling_I = np.where(I[i][j] == 0, 0.25, I[i][j])
                        up_sampling_I = up_sampling_I * (2**j)
                        up_sampling_I[0] = np.where(up_sampling_I[0] >= shape[i][0], up_sampling_I[0] - 2 ** (j-2), up_sampling_I[0])
                        up_sampling_I[1] = np.where(up_sampling_I[1] >= shape[i][1], up_sampling_I[1] - 2 ** (j-2), up_sampling_I[1])
                        up_sampling_I = np.round(up_sampling_I).astype(int)
                        index_writer.writerows(up_sampling_I)
                    else:
                        index_writer.writerows(I[i][j])
    return


#main
directory = sys.argv[1]
imgs_ori, P = readImages(directory)
print('read in images')

imgs = RGBtoGray(imgs_ori, P)

min_len = min(imgs[0].shape)
for i in range(P):
    min_len = min(min(imgs[i].shape), min_len)
S = 0
while min_len > 40:
    min_len /= 2
    S += 1

S = min(3, S)

#multi-scaling
multi_scaled_imgs = list()
for i in range(P):
    multi_scaled_imgs.append(funcs.multiScaling(imgs[i], S))
print('images multi-scaled')


#computing corner response
corner_responses = list()
for i in range(P):
    multi_scaled_corner_response = list()
    for j in range(S):
        multi_scaled_corner_response.append(cornerResponse(multi_scaled_imgs[i][j]))
    corner_responses.append(multi_scaled_corner_response)
print('corner respones computed')

#thresholding corner response
interest_points = list()
for i in range(P):
    thresholded_response = list()
    for j in range(S):
        thresholded_response.append( cornerResponseThresholding(corner_responses[i][j]) )
    interest_points.append(thresholded_response)
print('thresholded')

#sub-pixel refinement
offsets = list()
for i in range(P):
    multi_scaled_offset = list()
    for j in range(S):
        interest_points[i][j], offset = subPixelRefinement(interest_points[i][j], corner_responses[i][j])
        multi_scaled_offset.append(offset)
    offsets.append(multi_scaled_offset)
print('sub-pixel refinement done')

#non-max suppression
max_keypoints_num = 500
for i in range(P):
    for j in range(S):
        keypoints_num = math.ceil(max_keypoints_num / 4 ** j)
        interest_points[i][j], offsets[i][j] = nonMaximalSuppression(interest_points[i][j], corner_responses[i][j], keypoints_num, offsets[i][j])
print('non-maximal suppression done')

# #check interest points
# img_out = imgs[1].copy()
# m, n = imgs[0].shape
# for i in range(interest_points[1][0][0].size):
#     img_out[max(interest_points[1][0][0][i]-1, 0):min(interest_points[1][0][0][i] + 2, m), max(interest_points[1][0][1][i]-1, 0):min(interest_points[1][0][1][i]+2, n)] = 127
# cv.imwrite('check_interest_points_1.png', img_out)

#blurred gradient
orientations = list()
for i in range(P):
    multi_scaled_orientation = list()
    for j in range(S):
        multi_scaled_orientation.append(orientationAssignment(interest_points[i][j], corner_responses[i][j]))
    orientations.append(multi_scaled_orientation)
print('orientation assigned')

#discriptor vector
descriptors = list()
for i in range(P):
    multi_scaled_discriptor = list()
    for j in range(S):
        multi_scaled_discriptor.append(getDiscriptorVector(interest_points[i][j], orientations[i][j], offsets[i][j], multi_scaled_imgs[i][j]))
    descriptors.append(multi_scaled_discriptor)


#cylindrical projection
with open(f'{directory}/focal_length.csv', 'r') as focal_in_f:
    rows = csv.reader(focal_in_f)
    f = np.array(rows.__next__(), dtype=float)

imgs_p = []
interest_points_p = []
for i in range(P):
    img_p, interest_point_p = cproj.cylindricalProjection(interest_points[i], S, imgs_ori[i], f[i])
    imgs_p.append(img_p)
    interest_points_p.append(interest_point_p)

Path(f"{directory}/projected_img").mkdir(parents=True, exist_ok=True)
for i in range(P):
        cv.imwrite(f'{directory}/projected_img/{i}.png', imgs_p[i])

print('image projected')

#write descriptors to file
img_shape = [imgs[i].shape for i in range(P)]
writeDiscriptors(P, S, descriptors, interest_points, f'{directory}/descriptors.csv', f'{directory}/indices.csv', img_shape)
print(f'descriptors written to file {directory}/descriptors.csv')
