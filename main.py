import numpy as np
import multiprocessing as mp
import cv2

import librerias_patrones as lib_pat
from sklearn.metrics import confusion_matrix as CM


def extraction_routine(arr_name, lbp_grids, lbp_dists, har_grids, har_dists, gab_grids1, gab_grids2):
    """
    Saves to arr_name .npy the array of features extracted.
    Params MUST be tuples, use (1,) for single values
    :param lbp_grids: tuple
    :param lbp_dists: tuple, must be the same size than ibp_grids
    :param har_grids: idem
    :param har_dists: idem
    :param gab_grids1: idem
    :param gab_grids1: idem
    :return: void
    """
    names = lib_pat.get_img_names()
    images = [cv2.imread(names[i], 0) for i in range(len(names))]

    lbps = []
    for i in range(len(lbp_grids)):
        print('LPB Extraction, Iteration {}/{}'.format((i + 1), len(lbp_grids)))
        with mp.Pool() as p:
            lbp = p.starmap(lib_pat.get_LBP, [(images[j], lbp_dists[i], lbp_grids[i]) for j in range(len(images))])
        lbps.append(lbp)

    lbp_feats = np.concatenate(lbps, axis=1)

    hars = []
    for i in range(len(har_grids)):
        print('Haralick Extraction, Iteration {}/{}'.format((i + 1), len(har_grids)))
        with mp.Pool() as p:
            har = p.starmap(lib_pat.get_Haralick, [(images[j], har_dists[i], har_grids[i]) for j in range(len(images))])
        hars.append(har)
    har_feats = np.concatenate(hars, axis=1)

    gabs1 = []
    for i in range(len(gab_grids1)):
        print('Gabor Extraction, Iteration {}/{}'.format((i + 1), len(gab_grids1)))
        with mp.Pool() as p:
            gab = p.starmap(lib_pat.get_Gab, [(images[j], gab_grids1[i]) for j in range(len(images))])
        gabs1.append(gab)
    gab_feats1 = np.concatenate(gabs1, axis=1)

    gabs2 = []
    for i in range(len(gab_grids2)):
        print('Gabor Extraction, Iteration {}/{}'.format((i + 1), len(gab_grids2)))
        with mp.Pool() as p:
            gab = p.starmap(lib_pat.get_Gab_real_im, [(images[j], gab_grids2[i]) for j in range(len(images))])
        gabs2.append(gab)
    gab_feats2 = np.concatenate(gabs2, axis=1)

    feats = np.concatenate((lbp_feats, har_feats, gab_feats1, gab_feats2), axis=1)
    np.save(arr_name, feats)
    return feats


def generate_labels(lbp_grids, har_grids, gab_grids1, gab_grids2):
    # lbp 59
    # har 52
    # gab1 96
    # gab2 192
    c = 0
    labels = []
    for i in lbp_grids:
        for j in range(i ** 2):
            labels.append(c * np.ones(59, np.dtype(int)))
            c += 1
    for i in har_grids:
        for j in range(i ** 2):
            labels.append(c * np.ones(52, np.dtype(int)))
            c += 1
    for i in gab_grids1:
        for j in range(i ** 2):
            labels.append(c * np.ones(96, np.dtype(int)))
            c += 1
    for i in gab_grids2:
        for j in range(i ** 2):
            labels.append(c * np.ones(192, np.dtype(int)))
            c += 1
    lab = np.concatenate(labels)
    return lab


def red_routine_per_batch2(X, pca_ratio=.99):
    """
    Takes a batch of features and expands them with kpca and pca. then performs a RFECV selection.
    :param X:
    :param y:
    :param kpca_ratio:
    :param pca_ratio:
    :param cv_folds_num:
    :param rfecv_step:
    :return: X_tr, X_te, index
    """
    n = len(X[0])
    X_tr, X_te, y_tr, y_te = lib_pat.separate_train_test(X)
    pca_tr, pca_te = lib_pat.dim_red_auto_PCA(X_tr, X_te, pca_ratio)
    return pca_tr, pca_te


def reduction_routine(feats, labels, ratio=.99):
    """
    Reduces the number of features by appling PCA to every set of features for every window.
    :param feats:
    :param labels:
    :param ratio:
    :return:
    """
    f_tr = []
    f_te = []
    n = np.max(labels) + 1
    print('feature redution routine...')
    for i in range(n):
        print('Iteration {}/{}'.format(i, n))
        index = labels == i
        if np.count_nonzero(index) == 0:
            continue
        X = feats[:, index]
        tr, te = red_routine_per_batch2(X, ratio)
        f_tr.append(tr)
        f_te.append(te)
    X_tr = np.concatenate(f_tr, axis=1)
    X_te = np.concatenate(f_te, axis=1)
    print(X_tr.shape, X_te.shape)
    return X_tr, X_te


if __name__ == '__main__':
    """
    Comentando y descomentando se pueden 
    
    extraer = True, False
    true para la primera vez
    
    set = {1,2,3,4}
    
    
    """
    dataset = 4
    extraer = False

    print('Extracting and/or reading features...')
    if dataset == 1:
        lbp_params = ((1, 2, 2, 5), (5, 8, 15, 6))
        har_params = ((1, 1, 2, 5), ( ))
        gab1_params = (1,)
        gab2_params = (1, 5, 10)
        if extraer:
            extraction_routine('raw_features', lbp_params[0], lbp_params[1],
                               har_params[0], har_params[1], gab1_params, gab2_params)
        feats = np.load('raw_features.npy')
    elif dataset == 2:
        lbp_params = ((1, 2, 2, 5), (5, 8, 15, 6))
        har_params = ((1, 1, 2, 5), (1, 20, 11, 8))
        gab1_params = (1, 2, 5)
        gab2_params = (1, 2, 5)
        if extraer:
            extraction_routine('big_raw_features', lbp_params[0], lbp_params[1],
                               har_params[0], har_params[1], gab1_params, gab2_params)
        feats = np.load('big_raw_features.npy')
    elif dataset == 3:
        lbp_params = ((1, 2, 2, 5), (5, 8, 15, 6))
        har_params = ((1, 1, 2, 5), (1, 20, 11, 8))
        gab1_params = (1, 2, 5)
        gab2_params = (1, 2, 5, 10)
        if extraer:
            extraction_routine('huge_raw_features', lbp_params[0], lbp_params[1],
                               har_params[0], har_params[1], gab1_params, gab2_params)
        feats = np.load('huge_raw_features.npy')
    elif dataset == 4:
        lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
        har_params = ((1, 1, 1, 2, 5), (1, 10, 20, 11, 8))
        gab1_params = (1, 2, 5, 10)
        gab2_params = (1, 2, 5, 10)
        if extraer:
            extraction_routine('very_huge_raw_raw_features', lbp_params[0], lbp_params[1],
                               har_params[0], har_params[1], gab1_params, gab2_params)
        feats = np.load('very_huge_raw_features.npy')
    else:
        quit()

    labels = generate_labels(lbp_params[0], har_params[0], gab1_params, gab2_params)

    print('Removing features with low variance')
    feats, labels = lib_pat.delete_zero_variance_features(feats, labels, 0.1)
    print('Separating Features...')
    X_tr, X_te, y_tr, y_te = lib_pat.separate_train_test(feats)

    print('Reducing features by transformation')
    X_tr, X_te = reduction_routine(feats, labels, .99)
    print('Final reduction (for no colinear features)')
    X_tr, X_te = lib_pat.dim_red_auto_PCA(X_tr, X_te, ratio=.9)

    print('Classification via KNN 9')

    k1 = lib_pat.classification_knn(X_tr, X_te, y_tr, y_te, 9)

    print('Classification via SVC linear')
    k2 = lib_pat.classification_SVM(X_tr, X_te, y_tr, y_te, kernel='linear')

    print('Classification via SVC poli')
    k3 = lib_pat.classification_SVM(X_tr, X_te, y_tr, y_te, kernel='poly', degree=3)

    print('Classification via LDA solver=svd')
    k4 = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te, solver='svd')

    np.savetxt('k1', CM(y_te, k1), fmt='%2i', delimiter=',')
    np.savetxt('k2', CM(y_te, k2), fmt='%2i', delimiter=',')
    np.savetxt('k3', CM(y_te, k3), fmt='%2i', delimiter=',')
    np.savetxt('k4', CM(y_te, k4), fmt='%2i', delimiter=',')

    quit()
