
def featureMatching(P, S, D, cur):
    vote = np.zeros(P)
    matches = []
    K = 4
    for j in range(S):
        tree_data = np.zeros((0, 64))
        tree_data_img_index = []
        tree_data_descriptor_index = []
        for i in range(P):
            if i == cur:
                continue
            tree_data = np.r_[tree_data, D[i][j]]
            tree_data_img_index += [i] * D[i][j].shape[0]
            tree_data_descriptor_index += list(range(D[i][j].shape[0]))
        tree_data_img_index = np.array(tree_data_img_index, dtype=int)
        tree_data_descriptor_index = np.array(tree_data_descriptor_index, dtype=int)

        tree = KDTree(tree_data)
        dist, index = tree.query(D[cur][j], k=K)
        dist, index = np.array(dist).T, np.array(index).T

        for k in range(K):
            img_idx = tree_data_img_index[tuple([index[k], ])]
            imgs_match = [np.zeros((2, 0), dtype=int)] * P
            for p in range(P):
                d_idx = np.where(img_idx==p)
                if d_idx[0].size == 0:
                    continue
                d = 1 / (dist[k][d_idx] * (k+1))
                imgs_match[p] = np.c_[imgs_match[p], np.array([d_idx[0], tree_data_descriptor_index[d_idx] ])]
                vote[p] += d.sum()
                # vote[p] += d_idx[0].size 
        matches.append(imgs_match)
    
    # find the 6 images with most matches
    candidate = np.argsort(-vote)
    # print(np.sort(-vote))
    
    print(candidate)
    # potential matches
    candidate_matches = []
    for c in range(6):
        p = candidate[c]
        candidate_multi_scale_match = []
        for s in range(S):
            candidate_multi_scale_match.append(matches[s][p])
        candidate_matches.append(candidate_multi_scale_match)
    return candidate, candidate_matches

def imageMatching(P, S, I, D):
    # feature matching for each image
    candidates, candidates_match = [], []
    for i in range(P):
        candidate, candidate_match = featureMatching(P, S, D, i)
        candidates.append(candidate)
        candidates_match.append(candidate_match)

    #RANSAC
    correct_match = []
    correct_H = []
    for i in range(P):
        correct = []
        Hs = []
        for c in range(6):
            potential_matches = np.zeros((4, 0))
            p = candidates[i][c]
            matched_index = candidates_match[i][c]
            for s in range(S):
                self_x = np.array(I[i][s][0][tuple([matched_index[s][0]])]).reshape(1, -1)
                self_y = np.array(I[i][s][1][tuple([matched_index[s][0]])]).reshape(1, -1)
                next_x = np.array(I[p][s][0][tuple([matched_index[s][1],])]).reshape(1, -1)
                next_y = np.array(I[p][s][1][tuple([matched_index[s][1],])]).reshape(1, -1)
                potential_matches = np.concatenate((potential_matches, np.r_[self_x, self_y, next_x, next_y]), axis=1)
            inlier_index, H = RANSAC(potential_matches[0:2], potential_matches[2:4], thres=140)
            # matches = np.r_[potential_matches[0][inlier_index].reshape(1, -1), potential_matches[1][inlier_index].reshape(1, -1), potential_matches[2][inlier_index].reshape(1, -1), potential_matches[3][inlier_index].reshape(1, -1)]
            # plot_matches(matches.T, i, p)
            if inlier_index.size > 5.9 + 0.22 * potential_matches[0].size:
                correct.append(p)
                Hs.append(H)
        print(correct)
        correct_match.append(correct)
        correct_H.append(Hs)
    return correct_match, correct_H
