import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from GB_generation_with_idx import get_GB
from tqdm import tqdm


def search_natural_neighbor(X):
    n, m = X.shape
    r = 1 + 1
    kdtree = KDTree(X)
    nb = np.zeros(n)
    NN = {}
    RNN = {}
    for i in range(n):
        NN[i] = set()
        RNN[i] = set()
    num1 = -1
    num2 = 0
    while True:
        for idx, x in enumerate(X):
            _, idxs = kdtree.query(x, k=r)
            if r == 1:
                idxs = [idxs]
            y = idxs[-1]
            nb[y] = nb[y] + 1
            NN[idx].add(y)
            RNN[y].add(idx)
        num2 = np.sum(nb == 0)
        if num2 == num1:
            break
        else:
            num1 = num2
        r = r + 1
    return NN, RNN

def similarity(X, NaN, sigma):
    n, m = X.shape
    fs = np.zeros((n, n))
    for i in range(n):
        x_nn = X[list(NaN[i])]
        x_std = np.std(x_nn)
        for j in range(i + 1):
            y_nn = X[list(NaN[j])]
            y_std = np.std(y_nn)
            std_gap = abs(x_std - y_std)
            dist = np.linalg.norm(X[i] - X[j], ord=2) / m
            fs[i][j] = np.exp(-dist / (sigma * (1 - std_gap)))
            fs[j][i] = fs[i][j]
    return fs

def NGOD(X, GBs, NaN, sigma, n_data):
    n, m = X.shape
    attrs = np.arange(m)
    weight = np.zeros((n, m))
    approximation = np.zeros((n, m))
    for j, B in tqdm(enumerate(attrs), total=len(attrs)):
        fs_B = similarity(X[:, [B]], NaN, sigma)
        S = attrs
        fs_S = similarity(X[:, S], NaN, sigma)
        for i in range(n):
            low = np.zeros(n)
            up = np.zeros(n)
            for k in range(n):  # o
                cur_fs_B = fs_B[i][list(NaN[k])]
                cur_fs_S = fs_S[k][list(NaN[k])]
                up_appr = np.max(np.minimum(cur_fs_S, cur_fs_B))
                low_appr = np.min(np.maximum(1 - cur_fs_S, cur_fs_B))
                low[k] = low_appr
                up[k] = up_appr
            approximation[i, j] = np.sum(low) / np.sum(up)
            weight[i, j] = fs_B[i].mean()

    OD = 1 - approximation * weight
    GBOF = np.mean(OD * (1 - np.sqrt(weight)), axis=1) 
    
    OF = np.zeros(n_data)
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OF[point_idxs] = GBOF[idx]
        
    return OF


if __name__ == '__main__':
    data = pd.read_csv("./Example.csv")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    n, m = X.shape
    GBs = get_GB(X)
    n_gb = len(GBs)
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
    NN, RNN = search_natural_neighbor(centers)
    NaN = {}
    for i in range(n_gb):
        s = NN[i] & RNN[i]
        s.add(i)
        NaN[i] = s
    OF = NGOD(centers, GBs, NaN, 0.6, n)
    print(f"outlier factor: {OF}")
