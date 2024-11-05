import numpy as np
import pandas as pd
import cvxpy as cvx
import quadprog
from sklearn import metrics
from sklearn.metrics.pairwise import manhattan_distances


class Distances(object):

    def __init__(self, P, Q):
        if sum(P) < 1e-20 or sum(Q) < 1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P) != len(Q):
            raise "Arrays need to be of equal sizes..."
        # use numpy arrays for efficient coding
        P = np.array(P, dtype=float)
        Q = np.array(Q, dtype=float)
        # Correct for zero values
        P[np.where(P < 1e-20)] = 1e-20
        Q[np.where(Q < 1e-20)] = 1e-20
        self.P = P
        self.Q = Q

    def sqEuclidean(self):
        P = self.P
        Q = self.Q
        d = len(P)
        return sum((P - Q) ** 2)

    def probsymm(self):
        P = self.P
        Q = self.Q
        d = len(P)
        return 2 * sum((P - Q) ** 2 / (P + Q))

    def topsoe(self):
        P = self.P
        Q = self.Q
        return sum(P * np.log(2 * P / (P + Q)) + Q * np.log(2 * Q / (P + Q)))

    def hellinger(self):
        P = self.P
        Q = self.Q
        return 2 * np.sqrt(1 - sum(np.sqrt(P * Q)))


def DyS_distance(sc_1, sc_2, measure):
    dist = Distances(sc_1, sc_2)

    if measure == 'topsoe':
        return dist.topsoe()
    elif measure == 'probsymm':
        return dist.probsymm()
    elif measure == 'hellinger':
        return dist.hellinger()
    else:
        return 2


def TernarySearch(left, right, f, eps=1e-4):
    while True:
        if abs(left - right) < eps:
            return (left + right) / 2

        leftThird = left + (right - left) / 3
        rightThird = right - (right - left) / 3

        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird


def getHist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins) + 1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks, 1.1)

    re = np.repeat(1 / (len(breaks) - 1), (len(breaks) - 1))
    for i in range(1, len(breaks)):
        re[i - 1] = (re[i - 1] + len(np.where((scores >= breaks[i - 1]) & (scores < breaks[i]))[0])) / (len(scores) + 1)
    return re


def ACC(pred_labels, tpr, fpr):
    cc_ouput = round(pred_labels[pred_labels == 'True'].count() / len(pred_labels), 3)
    diff_tpr_fpr = (float(tpr) - float(fpr))
    one_prop = (cc_ouput - float(fpr)) / diff_tpr_fpr

    if one_prop <= 0:  # clipping the output between [0,1]
        pos_prop = 0
    elif one_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = one_prop

    return pos_prop


def HDy(pos_scores, neg_scores, test_scores):
    bin_size = np.linspace(10, 110, 11)  # creating bins from 10 to 110 with step size 10
    alpha_values = [round(x, 2) for x in np.linspace(0, 1, 101)]

    result = []
    num_bins = []
    for bins in bin_size:

        p_bin_count = getHist(pos_scores, bins)
        n_bin_count = getHist(neg_scores, bins)
        te_bin_count = getHist(test_scores, bins)

        vDist = []

        for x in range(0, len(alpha_values), 1):
            vDist.append(DyS_distance(((p_bin_count * alpha_values[x]) + (n_bin_count * (1 - alpha_values[x]))),
                                           te_bin_count, measure="hellinger"))

        result.append(alpha_values[np.argmin(vDist)])

    pos_prop = round(np.median(result), 2)
    return pos_prop


def DyS(pos_scores, neg_scores, test_scores, measure='topsoe'):
    bin_size = np.linspace(2, 20, 10)  # [10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)

    # print('bin_size', bin_size)
    result = []
    for bins in bin_size:
        p_bin_count = getHist(pos_scores, bins)
        n_bin_count = getHist(neg_scores, bins)
        te_bin_count = getHist(test_scores, bins)

        def f(x):
            return DyS_distance(((p_bin_count * x) + (n_bin_count * (1 - x))), te_bin_count, measure=measure)

        result.append(TernarySearch(0, 1, f))


    pos_prop = round(np.median(result), 4)
    return pos_prop


def GAC(y_hat, train_labels, yt_hat, classes):
    CM = metrics.confusion_matrix(train_labels, yt_hat, normalize="true").T

    df_p_y_hat = pd.DataFrame({cls: [0] for cls in classes})
    for pred_label in y_hat:
        df_p_y_hat[pred_label][0] = df_p_y_hat[pred_label][0]+1
    p_y_hat = df_p_y_hat.to_numpy()
    p_y_hat = np.squeeze(p_y_hat / p_y_hat.sum())

    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value


def GPAC(train_scores, test_scores, train_labels, classes):

    CM = np.zeros((len(classes), len(classes)))
    for cls in classes:
        idx = np.where(train_labels == cls)[0]
        i = classes.index(cls)
        CM[i] = np.sum(train_scores[idx], axis=0)
        CM[i] /= np.sum(CM[i])
    CM = CM.T
    p_y_hat = np.sum(test_scores, axis=0)
    p_y_hat = p_y_hat / np.sum(p_y_hat)

    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value


def EDy(tr_scores, labels, te_scores, classes):
    distance = manhattan_distances
    train_distrib_ = dict.fromkeys(classes)
    train_n_cls_i_ = np.zeros((len(classes), 1))

    if len(labels) == len(tr_scores):
        y_ext_ = labels
    else:
        y_ext_ = np.tile(labels, len(tr_scores) // len(labels))

    for n_cls, cls in enumerate(classes):
        train_distrib_[cls] = tr_scores[y_ext_ == cls, :]
        train_n_cls_i_[n_cls, 0] = len(train_distrib_[cls])

    K_, G_, C_, b_ = compute_ed_param_train(distance, train_distrib_, classes, train_n_cls_i_)

    a_ = compute_ed_param_test(distance, train_distrib_, te_scores, K_, classes, train_n_cls_i_)

    prevalences = solve_ed(G=G_, a=a_, C=C_, b=b_)

    return prevalences / np.sum(prevalences)


def dpofa(m):
    r = np.array(m, copy=True)
    n = len(r)
    for k in range(n):
        s = 0.0
        if k >= 1:
            for i in range(k):
                t = r[i, k]
                if i > 0:
                    t = t - np.sum(r[0:i, i] * r[0:i, k])
                t = t / r[i, i]
                r[i, k] = t
                s = s + t * t
        s = r[k, k] - s
        if s <= 0.0:
            return k+1, r
        r[k, k] = np.sqrt(s)
    return 0, r


def nearest_pd(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    indendity_matrix = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += indendity_matrix * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def is_pd(m):
    return dpofa(m)[0] == 0


def solve_ed(G, a, C, b):
    sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
    prevalences = sol[0]
    # the last class was removed from the problem, its prevalence is 1 - the sum of prevalences for the other classes
    return np.append(prevalences, 1 - prevalences.sum())


def compute_ed_param_train(distance_func, train_distrib, classes, n_cls_i):
    n_classes = len(classes)
    #  computing sum de distances for each pair of classes
    K = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        K[i, i] = distance_func(train_distrib[classes[i]], train_distrib[classes[i]]).sum()
        for j in range(i + 1, n_classes):
            K[i, j] = distance_func(train_distrib[classes[i]], train_distrib[classes[j]]).sum()
            K[j, i] = K[i, j]

    #  average distance
    K = K / np.dot(n_cls_i, n_cls_i.T)

    B = np.zeros((n_classes - 1, n_classes - 1))
    for i in range(n_classes - 1):
        B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
        for j in range(n_classes - 1):
            if j == i:
                continue
            B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

    #  computing the terms for the optimization problem
    G = 2 * B
    if not is_pd(G):
        G = nearest_pd(G)

    C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
    b = -np.array([1] + [0] * (n_classes - 1), dtype=float)

    return K, G, C, b


def compute_ed_param_test(distance_func, train_distrib, test_distrib, K, classes, n_cls_i):

    n_classes = len(classes)
    Kt = np.zeros(n_classes)
    for i in range(n_classes):
        Kt[i] = distance_func(train_distrib[classes[i]], test_distrib).sum()

    Kt = Kt / (n_cls_i.squeeze() * float(len(test_distrib)))

    a = 2 * (- Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])
    return a


def EMQ(test_scores, nclasses, p_tr):
    max_it = 1000
    eps = 1e-6

    p_s = np.copy(p_tr)
    p_cond_tr = np.array(test_scores)
    p_cond_s = np.zeros(p_cond_tr.shape)

    for it in range(max_it):
        r = p_s / p_tr
        p_cond_s = p_cond_tr * r
        s = np.sum(p_cond_s, axis=1)
        for c in range(nclasses):
            p_cond_s[:, c] = p_cond_s[:, c] / s
        p_s_old = np.copy(p_s)
        p_s = np.sum(p_cond_s, axis=0) / p_cond_s.shape[0]
        if (np.sum(np.abs(p_s - p_s_old)) < eps):
            break

    return p_s/np.sum(p_s)
