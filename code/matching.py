import numpy as np
import sys
import math
import csv
from scipy.spatial import KDTree
import random
import matplotlib.pyplot as plt
import cv2 as cv

def readFeatureData(directory):
    I = list()
    D = list()
    with open(f'{directory}/indices.csv', 'r') as index_f:
        rows = csv.reader(index_f)
        row = rows.__next__()
        P, S = int(row[0]), int(row[1])
        for i in range(P):
            multi_scaled_interest_point = list()
            for j in range(S):
                pos_x = np.array(rows.__next__(), dtype=float)
                pos_y = np.array(rows.__next__(), dtype=float)
                if pos_x[0] == -1:
                    pos_x = np.zeros((0, 64))
                    pos_y = np.zeros((0, 64))
                interest_point = tuple([pos_x, pos_y])
                multi_scaled_interest_point.append(interest_point)
            I.append(multi_scaled_interest_point)
        
    with open(f'{directory}/descriptors.csv', 'r') as dis_f:
        rows = csv.reader(dis_f)
        for i in range(P):
            multi_scaled_descriptor = list()
            for j in range(S):
                descriptor = list()
                for k in range(I[i][j][0].size):
                    descriptor.append(np.array(rows.__next__(), dtype=float))
                multi_scaled_descriptor.append(np.array(descriptor))
            D.append(multi_scaled_descriptor)
    return P, S, I, D

def computeErrors(pos_1, pos_2, H):
    pos_estimated = np.array([pos_2[0] + H[0], pos_2[1] + H[1]])
    errors = ((pos_1 - pos_estimated) ** 2).sum(axis = 0)
    errors = errors ** 0.5
    return errors


def RANSAC(pos_1, pos_2, iter = 1000, thres = 10):
    if pos_1[0].size < 1:
        return np.zeros(0), np.zeros((3, 3))
    best_model_vote = -1
    best_model_inliers = np.zeros(0)
    best_H = np.zeros((2), dtype=int)
    total = pos_1[0].size
    for it in range(iter):
        #sampling points
        r = random.sample(range(total), 1)
        H = np.array([pos_1[0][r] - pos_2[0][r], pos_1[1][r] - pos_2[1][r]])
        #compute error
        errors = computeErrors(pos_1, pos_2, H)
        # print(np.sort(errors))
        #thresholding
        
        inliers = np.where(errors < thres)[0]

        vote = inliers.size
        if vote > best_model_vote:
            best_model_vote = vote
            best_model_inliers = inliers
            best_H = H
    return best_model_inliers, best_H.reshape((1, 2))

def matchVerification(I, D, candidate, candidate_index, cur):
    inliers, Hs = [], []
    Hs.append(np.zeros((1, 2), dtype=int))
    for i in range(6):
        matched_index_cur = tuple([candidate_index[i][0].astype(int)])
        matched_index_can = tuple([candidate_index[i][1].astype(int)])
        matched_pos_cur = np.array([I[cur][0][0][matched_index_cur], I[cur][0][1][matched_index_cur]])
        matched_pos_can = np.array([I[candidate[i]][0][0][matched_index_can], I[candidate[i]][0][1][matched_index_can]])
        inlier, H = RANSAC(matched_pos_cur, matched_pos_can)
        inliers.append(inlier)
        Hs.append(H)
    return

def featureMatching(D1, D2):
    matched_index = []
    for s in range(S):
        tree = KDTree(D2[s])
        dist, index = tree.query(D1[s], k=2)
        dist, index = np.array(dist).T, np.array(index).T
        NN_1_dist = dist[0]
        NN_2_dist = dist[1]
        NN_1_index = index[0]
        self_matched = np.where(NN_1_dist < NN_2_dist * 0.75)
        matched = NN_1_index[self_matched]
        matched_index.append(np.r_[np.array(self_matched[0]).reshape(1, -1), matched.reshape(1, -1)])
    return matched_index

def writeMatches(P, H, directory):
    # H = np.array(H).reshape(P-1, 9)
    with open(f'{directory}/H.csv', 'w') as H_out:
        writer = csv.writer(H_out)
        writer.writerows(H)
    return

def imageMatching(P, S, I, D):
    all_potential_matches = []
    inliers, Hs = [], []
    Hs.append(np.zeros((1, 2), dtype=int))
    for p in range(P-1):
        matched_index = featureMatching(D[p], D[p+1])
        potential_matches = np.zeros((4, 0))

        for s in range(S):
            self_x = np.array(I[p][s][0][tuple([matched_index[s][0]])]).reshape(1, -1)
            self_y = np.array(I[p][s][1][tuple([matched_index[s][0],])]).reshape(1, -1)
            next_x = np.array(I[p+1][s][0][tuple([matched_index[s][1],])]).reshape(1, -1)
            next_y = np.array(I[p+1][s][1][tuple([matched_index[s][1],])]).reshape(1, -1)
            potential_matches = np.concatenate((potential_matches, np.r_[self_x, self_y, next_x, next_y]), axis=1)
        all_potential_matches.append(potential_matches)
        inlier_index, H = RANSAC(potential_matches[0:2], potential_matches[2:4])
        inliers.append(inlier_index)
        Hs.append(H)
    Hs = np.array(Hs, dtype=int).reshape((P, 2))
    return Hs
 

#read in indices and descriptors 
directory = sys.argv[1]
P, S, interest_points, descriptors = readFeatureData(directory)

H = imageMatching(P, S, interest_points, descriptors)
writeMatches(P, H, directory)

