import time

import multiprocessing as mp
import numpy as np

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.feature_selection import RFECV, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

import skimage
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage import io

from mahotas.features import haralick, pftas
import mahotas

import cv2

debug = True


def get_LBP(image_array, radius=1, grid_size=1, j=10):
    """
    LBP para una imagen. 59 bins
    :param image_array:
    :param radius:
    :param grid_size:
    :return:
    """
    if j % 50 == 0: 
        print("|", end = "", flush = True)
    p = 8
    img = np.asarray(image_array)
    window_size = (np.asarray([img.shape]) / grid_size).astype(int)[0]
    im_grid = np.asarray(skimage.util.view_as_blocks(img, tuple(window_size)))
    windows = []
    for i in range(grid_size):
        for j in range(grid_size):
            windows.append(im_grid[i, j])

    lbp_features = []
    for i in range(len(windows)):
        hist = np.histogram(local_binary_pattern(windows[i], p, radius, method='nri_uniform'), bins=59, density=True)[0]
        lbp_features.append(hist)
    out = np.ravel(np.asarray(lbp_features))
    return out


def get_Haralick(im_arr, dist=1, grid_size=1, j=10):
    """
    Haralick para una imagen.
    :param im_arr:
    :param dist:
    :param grid_size:
    :return: 13*4*grid_size^2 array length
    """

    if j % 50 == 0: 
        print("|", end = "", flush = True)
    img = np.asarray(im_arr).astype(int)
    img = mahotas.stretch(img, 31)
    window_size = (np.asarray([img.shape]) / grid_size).astype(int)[0]
    im_grid = np.asarray(skimage.util.view_as_blocks(img, tuple(window_size)))
    windows = []
    for i in range(grid_size):
        for j in range(grid_size):
            windows.append(im_grid[i, j])
    haralick_features = []
    for i in range(len(windows)):
        h = haralick(windows[i], distance=dist)
        h = np.ravel(np.asarray(h))
        haralick_features.append(h)
    out = np.ravel(np.asarray(haralick_features))
    return out


def get_Gab(img_array, grid_size=1, j=10):
    """
    Gabor filters.
    :param im_path:
    :param grid_size:
    :return:
    """
    if j % 50 == 0: 
        print("|", end = "", flush = True)
    img = np.asarray(img_array)
    window_size = (np.asarray([img.shape]) / grid_size).astype(int)[0]
    im_grid = np.asarray(skimage.util.view_as_blocks(img, tuple(window_size)))
    windows = []
    for i in range(grid_size):
        for j in range(grid_size):
            windows.append(im_grid[i, j])
    freq_step = np.sqrt(2)
    or_step = np.pi / 4.
    gab_feat = []
    for i in range(len(windows)):
        for j in range(6):  # frequencies
            for k in range(8):  # orientations
                a = gabor(windows[i], frequency=(.25 / (freq_step ** j)), theta=k * or_step, sigma_x=1,
                          sigma_y=1)
                b = np.sqrt(a[0] ** 2 + a[1] ** 2)
                mean = np.mean(b)
                std = np.std(b)
                if mean == np.inf: mean = 0
                if std == np.inf: std = 0
                gab_feat.append(mean)
                gab_feat.append(std)
    out = np.ravel(np.asarray(gab_feat))
    return out


def get_Gab_real_im(img_array, grid_size=1, j=10):
    """
    Gabor filters, not combining real an imaginary parts
    :param im_path:
    :param grid_size:
    :return:
    """
    if j % 50 == 0: 
        print("|", end = "", flush = True)
    img = np.asarray(img_array)
    window_size = (np.asarray([img.shape]) / grid_size).astype(int)[0]
    im_grid = np.asarray(skimage.util.view_as_blocks(img, tuple(window_size)))
    windows = []
    for i in range(grid_size):
        for j in range(grid_size):
            windows.append(im_grid[i, j])
    freq_step = np.sqrt(2)
    or_step = np.pi / 4.
    gab_feat = []
    for i in range(len(windows)):
        for j in range(6):  # frequencies
            for k in range(8):  # orientations
                a = gabor(windows[i], frequency=(.25 / (freq_step ** j)), theta=k * or_step, sigma_x=1,
                          sigma_y=1)
                mean1 = np.mean(a[0])
                mean2 = np.mean(a[1])
                std1 = np.std(a[0])
                std2 = np.std(a[1])
                if mean1 == np.inf: mean1 = 0
                if mean2 == np.inf: mean2 = 0
                if std1 == np.inf: std1 = 0
                if std2 == np.inf: std2 = 0
                gab_feat.append(mean1)
                gab_feat.append(mean2)
                gab_feat.append(std1)
                gab_feat.append(std2)
    out = np.ravel(np.asarray(gab_feat))
    return out


def get_TAS(im_path, grid_size=4):
    """
    Parameterless TAS para una imagen.
    :param im_path:
    :param grid_size:
    :return: 27* grid_size^2 (check)
    """
    img = mahotas.imread(im_path)
    window_size = (np.asarray([img.shape])[0:2] / grid_size).astype(int)[0]
    im_grid = np.asarray(skimage.util.view_as_blocks(img, tuple([window_size[0], window_size[1], 3])))
    windows = []
    for i in range(grid_size):
        for j in range(grid_size):
            windows.append(im_grid[i, j, 0])
    tas_features = []
    for i in range(len(windows)):
        if (i % 10 == 0):
            print("|", end = "", flush = True)
        t = pftas(windows[i])
        t = np.ravel(np.asarray(t))
        tas_features.append(t)
    print(" ")
    out = np.ravel(np.asarray(tas_features))
    return out


def get_HoG(im_path, grid_size=4):
    """
    Histogram of Gradients para una imagen. Low level, quizás no debería usar...
    :param im_path:
    :param grid_size:
    :return:
    """
    img_array = io.imread(im_path, True)
    img = np.asarray(img_array)
    window_size = (np.asarray([img.shape]) / grid_size).astype(int)[0]
    im_grid = np.asarray(skimage.util.view_as_blocks(img, tuple(window_size)))
    windows = []
    # print(im_grid[0, 0].shape)
    for i in range(grid_size):
        for j in range(grid_size):
            windows.append(im_grid[i, j])

    hog_features = []
    for i in range(len(windows)):
        if (i % 10 == 0):
            print("|", end = "", flush = True)
        hist = hog(windows[i])
        hog_features.append(hist)
    out = np.ravel(np.asarray(hog_features))
    print(" ")
    return out


def dim_red_PCA(tr_feat_matrix, te_feat_matrix, num_feat=100):
    """
    PCA dimensionality reduction, fixed amount of features.
    :param tr_feat_matrix:
    :param te_feat_matrix:
    :param num_feat:
    :return:
    """
    pca = PCA(n_components=num_feat)
    pca.fit(tr_feat_matrix)
    training_matrix = pca.transform(tr_feat_matrix)
    testing_matrix = pca.transform(te_feat_matrix)
    return training_matrix, testing_matrix


def dim_red_auto_PCA(tr_feat_matrix, te_feat_matrix, ratio=.995):
    """
    PCA dimensionality reduction, fixed explained variance.
    :param tr_feat_matrix:
    :param te_feat_matrix:
    :param ratio:
    :return:
    """
    pca = PCA()
    pca.fit(tr_feat_matrix)

    exp = 0.
    count = 0
    for i in range(len(tr_feat_matrix)):
        exp += pca.explained_variance_ratio_[i]
        if exp >= ratio:
            count = i + 1
            break
    training_matrix = np.asarray(pca.transform(tr_feat_matrix))[:, 0:count]
    testing_matrix = np.asarray(pca.transform(te_feat_matrix))[:, 0:count]
    return training_matrix, testing_matrix


def dim_red_KPCA(tr_feat_matrix, te_feat_matrix, num_feat=100, ker='cosine', gamma=-1):
    """
    KPCA dimensionality reduction, fixed number of features.
    :param tr_feat_matrix:
    :param te_feat_matrix:
    :param num_feat:
    :return:
    """
    if gamma <= 0:
        kpca = KernelPCA(kernel=ker, n_components=num_feat)
    else:
        kpca = KernelPCA(kernel=ker, n_components=num_feat, gamma=gamma)
    training_matrix = kpca.fit_transform(tr_feat_matrix)
    testing_matrix = kpca.transform(te_feat_matrix)
    return training_matrix, testing_matrix


def dim_red_auto_KPCA(tr, te, ratio=.99, verbose=False, ker='cosine'):
    """
    KPCA dimensionality reduction, fixed explained variance (through eigenvalues sum ratio)
    :param tr:
    :param te:
    :param ratio:
    :param verbose:
    :return:
    """
    kpca = KernelPCA(kernel=ker, n_jobs=1)
    tr_out = kpca.fit_transform(tr)
    te_out = kpca.transform(te)
    t = np.sum(kpca.lambdas_)
    aux = 0
    index = 0
    for i in range(len(kpca.lambdas_)):
        if aux / t >= ratio:
            index = i
            if verbose: print('yeah!', index)
            break
        aux += kpca.lambdas_[i]
        index = i

    if verbose:
        for i in kpca.lambdas_:
            print(i)

    return tr_out[:, 0:index], te_out[:, 0:index]


def dim_red_TSDV(tr_feat_matrix, te_feat_matrix, num_feat=100):
    """
    Dimensionality reduction via truncated singular value decomposition aka. Latent semantic analysis
    :param tr_feat_matrix:
    :param te_feat_matrix:
    :param num_feat:
    :return:
    """
    tsdv = TruncatedSVD(n_components=num_feat)
    training = tsdv.fit_transform(tr_feat_matrix)
    testing = tsdv.transform(te_feat_matrix)
    return training, testing


def dim_red_auto_TSDV(tr_feat_matrix, te_feat_matrix, ratio=.99):
    """
    Dimensionality reducticion via truncated singular value decomposition. aka latent semantics analysis
    :param tr_feat_matrix:
    :param te_feat_matrix:
    :param ratio:
    :return:
    """
    tsdv = TruncatedSVD(n_components=len(tr_feat_matrix[0]) - 1)
    training = tsdv.fit_transform(tr_feat_matrix)
    exp = 0.
    count = 0
    for i in range(len(tr_feat_matrix)):
        exp += tsdv.explained_variance_ratio_[i]
        if exp >= ratio:
            count = i + 1
            break

    tsdv = TruncatedSVD(n_components=count)
    training = tsdv.fit_transform(tr_feat_matrix)
    testing = tsdv.transform(te_feat_matrix)
    return training, testing


def select_SFS(X_tr, y_tr, num_feat=100, knn_parameter=1, forward_=False, floating_=True):
    """
    Secuential Feature Selection
    :param X_tr:
    :param y_tr:
    :param num_feat:
    :param knn_parameter:
    :param forward_:
    :param floating_:
    :return:
    """
    X = X_tr
    y = y_tr
    knn = KNeighborsClassifier(n_neighbors=knn_parameter)
    sfs1 = SFS(knn,
               k_features=(1, num_feat),
               forward=forward_,
               floating=floating_,
               verbose=1,
               scoring='accuracy',
               cv=3,
               n_jobs=4)
    sfs1 = sfs1.fit(X, y)
    out = sfs1.k_feature_idx_
    return np.asarray(out)


def select_RFECV(X_tr, y_tr, k, step_=1):
    """
    Recursive feature elimination and cross-validated selection
    :param X_tr: training set
    :param y_tr: target
    :param estimator: object to score
    :param k: number of folds
    :param step_:
    :return:
    """
    X = X_tr
    y = y_tr
    estimator = SVC(kernel='linear')
    estimator.fit(X, y)
    print('bla')
    rfecv = RFECV(estimator, step=step_, n_jobs=4, cv=k, verbose=1, scoring='accuracy')
    print('ble')
    rfecv.fit(X, y)
    return np.nonzero(np.asarray(rfecv.ranking_) == 1)[0]


def select_secuential_RFECV(tr):
    index = select_RFECV(tr, 1000)
    index = select_RFECV(tr[:, index], 100)
    index = select_RFECV(tr[:, index], 10)
    return index


def select_RFECV_scoring(tr, step, num_cicles):
    """
    Scores training features based on how many times they got selected by RFECV
    :param tr:
    :param step:
    :param num_cicles:
    :return:
    """
    scores = np.zeros(len(tr[0]), dtype=int)
    print('Starting Scoring')
    for i in range(num_cicles):
        print('iter num:', i + 1, '/', num_cicles)
        spam = select_RFECV(tr, step)
        for j in spam:
            scores[j] += 1
    return scores


def select_RFECV_scoring_return_best(tr, te, max_num_feat, scores):
    c = np.max(scores)
    resp = np.zeros(len(scores), dtype=int)
    for i in range(c):
        curr = c - i
        index = np.nonzero(scores == curr)
        if len(np.union1d(index, resp)) < max_num_feat:
            resp = np.union1d(index, resp)
        else:
            dif = np.intersect1d(np.setdiff1d(index, resp), resp)
            n = max_num_feat - len(resp)
            resp = np.union1d(resp, dif[:n])
            break
    return tr[:, resp], te[:, resp]


def select_RFECV_scoring_return_frac(tr, te, frac, scores):
    c = np.max(scores)
    resp = np.zeros(len(scores), dtype=int)
    for i in range(c):
        curr = c - i
        index = np.nonzero(scores == curr)
        if len(np.union1d(index, resp)) < int(np.ceil(frac * len(scores))):
            resp = np.union1d(index, resp)
        else:
            dif = np.intersect1d(np.setdiff1d(index, resp), resp)
            n = int(np.ceil(frac * len(scores))) - len(resp)
            resp = np.union1d(resp, dif[:n])
            break
    return tr[:, resp], te[:, resp]


def select_iterative_RFECV(tr, step, iters):
    scores = select_RFECV_scoring(tr, step, iters)
    return np.nonzero(scores != 0)[0]


def select_KBest(training, num_feat=100, mutual_info_classif_=False):
    X = training
    y = [i for i in range(9) for j in range(60)]
    if mutual_info_classif_:
        c = mutual_info_classif
    else:
        c = f_classif
    SKB = SelectKBest(c, num_feat)
    SKB.fit(X, y)
    return SKB.get_support(True)


def select_KBest_trees(training, num_feat=100, n_est_trees=15, num_runs=10):
    X = training
    y = [i for i in range(9) for j in range(60)]
    scores = np.zeros(len(training[0]), dtype=float)
    for i in range(num_runs):
        print('fitting trees iter num:', i + 1, '/', num_runs)
        c = ExtraTreesClassifier(n_estimators=n_est_trees)
        c.fit(X, y)
        scores += c.feature_importances_
    support = np.zeros(len(training[0]))
    mn = np.min(scores)
    for i in range(num_feat):
        m = np.argmax(scores)
        support[m] = 1
        scores[m] = mn
    index = np.nonzero(support == 1)[0]
    return index


def select_EFS(tr, num_feat=100):
    X = tr
    y = [i for i in range(9) for j in range(60)]
    knn = KNeighborsClassifier(n_neighbors=1)
    efs = EFS(knn, min_features=num_feat, max_features=num_feat, cv=5, n_jobs=4)
    efs.fit(X, y)
    out = efs.best_idx_
    return out


def feat_extraction_routine():
    lbp_grid1 = 4
    lbp_dist1 = 10
    lbp_grid2 = 7
    lbp_dist2 = 3
    lbp_grid3 = 2
    lbp_dist3 = 19

    har_grid1 = 4
    har_dist1 = 2
    har_grid2 = 2
    har_dist2 = 15
    har_grid3 = 1
    har_dist3 = 4

    tas_grid1 = 2
    tas_grid2 = 4

    hog_grid = 2

    gab_grid1 = 4
    gab_grid2 = 7

    # get images
    names = [('fotos/face_' + str(i + 1).zfill(2) + '_' + str(j + 1).zfill(2) + '.png')
             for i in range(60) for j in range(10)]

    im_arrays = []

    for i in range(len(names)):
        im_arrays.append(np.asarray(mahotas.imread(names[i], True)))

    # im_arrays = np.asarray(im_arrays)
    labels = np.array([])
    labels2 = np.array([])
    counter = 0
    with mp.Pool() as p:
        tt = time.time()
        if debug: print("Extracting LBP1")
        f_lbp1 = p.starmap(get_LBP, [(i, lbp_dist1, lbp_grid1) for i in im_arrays])
        u1 = [i for i in range(lbp_grid1 ** 2) for j in range(int(len(f_lbp1[0]) / lbp_grid1 ** 2))]
        v1 = np.ones(len(f_lbp1[0]), dtype=int) * 0
        if debug: print('LBP time: ', str(time.time() - tt))
        tt = time.time()
        if debug: print("Extracting LBP2")
        f_lbp2 = p.starmap(get_LBP, [(i, lbp_dist2, lbp_grid2) for i in im_arrays])
        counter = np.max(u1) + 1
        u2 = ([(counter + i) for i in range(lbp_grid2 ** 2) for j in range(int(len(f_lbp2[0]) / lbp_grid2 ** 2))])
        v2 = np.ones(len(f_lbp2[0]), dtype=int) * 1
        if debug: print('LBP time: ', str(time.time() - tt))
        tt = time.time()
        if debug: print("Extracting LBP3")
        f_lbp3 = p.starmap(get_LBP, [(i, lbp_dist3, lbp_grid3) for i in im_arrays])
        counter = np.max(u2) + 1
        u3 = ([(counter + i) for i in range(lbp_grid3 ** 2) for j in range(int(len(f_lbp3[0]) / lbp_grid3 ** 2))])
        v3 = np.ones(len(f_lbp3[0]), dtype=int) * 2
        if debug: print('LBP time: ', str(time.time() - tt))

        tt = time.time()
        if debug: print("Extracting Haralick1")
        f_har1 = p.starmap(get_Haralick, [(i, har_dist1, har_grid1) for i in im_arrays])
        counter = np.max(u3) + 1
        u4 = ([(counter + i) for i in range(har_grid1 ** 2) for j in range(int(len(f_har1[0]) / har_grid1 ** 2))])
        v4 = np.ones(len(f_har1[0]), dtype=int) * 3
        if debug: print('Haralick time: ', str(time.time() - tt))
        tt = time.time()
        if debug: print("Extracting Haralick2")
        f_har2 = p.starmap(get_Haralick, [(i, har_dist2, har_grid2) for i in im_arrays])
        counter = np.max(u4) + 1
        u5 = ([(counter + i) for i in range(har_grid2 ** 2) for j in range(int(len(f_har2[0]) / har_grid2 ** 2))])
        v5 = np.ones(len(f_har2[0]), dtype=int) * 4
        if debug: print('Haralick time: ', str(time.time() - tt))
        tt = time.time()
        if debug: print("Extracting Haralick3")
        f_har3 = p.starmap(get_Haralick, [(i, har_dist3, har_grid3) for i in im_arrays])
        counter = np.max(u5) + 1
        u6 = ([(counter + i) for i in range(har_grid3 ** 2) for j in range(int(len(f_har3[0]) / har_grid3 ** 2))])
        v6 = np.ones(len(f_har3[0]), dtype=int) * 5
        if debug: print('Haralick time: ', str(time.time() - tt))

        tt = time.time()
        if debug: print("Extracting TAS1")
        f_tas1 = p.starmap(get_TAS, [(i, tas_grid1) for i in names])
        counter = np.max(u6) + 1
        u7 = ([(counter + i) for i in range(tas_grid1 ** 2) for j in range(int(len(f_tas1[0]) / tas_grid1 ** 2))])
        v7 = np.ones(len(f_tas1[0]), dtype=int) * 6
        if debug: print('TAS time: ', str(time.time() - tt))
        tt = time.time()
        if debug: print("Extracting TAS2")
        f_tas2 = p.starmap(get_TAS, [(i, tas_grid2) for i in names])
        counter = np.max(u7) + 1
        u8 = ([(counter + i) for i in range(tas_grid2 ** 2) for j in range(int(len(f_tas2[0]) / tas_grid2 ** 2))])
        v8 = np.ones(len(f_tas2[0]), dtype=int) * 7
        if debug: print('TAS time: ', str(time.time() - tt))

        tt = time.time()
        if debug: print("Extracting Gabor1")
        f_gab1 = p.starmap(get_Gab, [(i, gab_grid1) for i in names])
        counter = np.max(u8) + 1
        u9 = ([(counter + i) for i in range(gab_grid1 ** 2) for j in range(int(len(f_gab1[0]) / gab_grid1 ** 2))])
        v9 = np.ones(len(f_gab1[0]), dtype=int) * 8
        if debug: print('Gab time: ', str(time.time() - tt))
        tt = time.time()
        if debug: print("Extracting Gabor2")
        f_gab2 = p.starmap(get_Gab, [(i, gab_grid2) for i in names])
        counter = np.max(u9) + 1
        u10 = ([(counter + i) for i in range(gab_grid2 ** 2) for j in range(int(len(f_gab2[0]) / gab_grid2 ** 2))])
        v10 = np.ones(len(f_gab2[0]), dtype=int) * 9
        if debug: print('Gab time: ', str(time.time() - tt))

        tt = time.time()
        if debug: print("Extracting HoG")
        f_hog = p.starmap(get_HoG, [(i, hog_grid) for i in names])
        counter = np.max(u10) + 1
        u11 = ([(counter + i) for i in range(hog_grid ** 2) for j in range(int(len(f_hog[0]) / hog_grid ** 2))])
        v11 = np.ones(len(f_hog[0]), dtype=int) * 10
        if debug: print('Hog time: ', str(time.time() - tt))
    labels = np.concatenate((u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11))
    labels2 = np.concatenate((v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))
    l = np.asarray([labels, labels2])
    features = []

    for i in range(len(im_arrays)):
        features.append(np.concatenate((f_lbp1[i], f_lbp2[i], f_lbp3[i], f_har1[i], f_har2[i],
                                        f_har3[i], f_tas1[i], f_tas2[i], f_gab1[i], f_gab2[i],
                                        f_hog[i])).astype('float32'))
    return np.asarray(features), l


def feature_normalization(f):
    """
    Normalize Feature Matrix

    :param f: Feature Matrix
    :return: Normalized Feature Matrix
    """
    f = np.asarray(f)
    for i in range(len(f[0])):
        f[:, i] = (f[:, i] - np.mean(f[:, i])) / (np.std(f[:, i]) + 0.00001)
    return f


def delete_zero_variance_features(f, l, tol):
    """
    Removes features with std below threshold
    :param f: features matrix
    :param l: labels
    :param tol: float
    :return: f, l
    """
    index = np.std(f, axis=0) > tol
    print(np.count_nonzero(index == False), "features removed.")
    return f[:, index], l[index]


def feature_variance_trim(f, l, r_to_trim):
    """
    Trims the r% of features of less variance

    :param f: Feature Matrix
    :param l: labels
    :param r_to_trim: Ratio to be trimmed
    :return: Feature matrix trimmed in its original order
    """
    r = r_to_trim
    vars = np.std(f, axis=0)
    index = np.argsort(vars)
    to_trim = int(len(vars) * r)
    index = index[to_trim:]
    index = np.sort(index)
    return f[:, index], l[:, index]


def classification_knn(X_tr, X_te, y_tr, y_te, neighbors_param=3):
    """
    KNN classification
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :param neighbors_param:
    :return:
    """
    knn = KNeighborsClassifier(n_neighbors=neighbors_param, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    results = knn.predict(X_te)
    print('accuracy: ', knn.score(X_te, y_te))
    return results


def classification_SVM(X_tr, X_te, y_tr, y_te, kernel='linear', _C=1, degree=3):
    """
    Support Vector Machine Classificator, RBF kernel
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :return: prediction for X_te
    """
    if kernel == 'poly':
        svm = SVC(kernel=kernel, C=_C, degree=degree)
    else:
        svm = SVC(kernel=kernel, C=_C)
    svm.fit(X_tr, y_tr)
    print('accuracy:', svm.score(X_te, y_te))
    return svm.predict(X_te)


def classification_LDA(X_tr, X_te, y_tr, y_te, solver='lsqr'):
    """
    Linear Discriminant Analysis classification
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :param solver:
    :return:
    """
    if solver == 'svd':
        lda = LDA(solver=solver)
    else:
        lda = LDA(solver=solver, shrinkage='auto')
    lda.fit(X_tr, y_tr)
    results = lda.predict(X_te)
    print('accuracy:', lda.score(X_te, y_te))
    return results

def classification_MLP(X_tr, X_te, y_tr, y_te, solver='lbfgs'):
    """
    Multi-layer Perceptron classifier
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :param solver:
    :return:
    """
    
    # mlp = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp = MLPClassifier(hidden_layer_sizes=(50, ), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mlp.fit(X_tr, y_tr)
    results = mlp.predict(X_te)
    print('accuracy:', mlp.score(X_te, y_te))
    return results



# ----------------------------------------------------------------------------------------------------------------------


def separate_train_test(feats):
    """
    Separates the database in training and testing groups. Also labels the pictures.
    :param feats: Feature matrix
    :return: X_train, X_test, y_train, y_test
    """
    train = np.array([i for i in range(240 * 7)], np.dtype(int)) % 240 < 200
    test = np.array([i for i in range(240 * 7)], np.dtype(int)) % 240 >= 200
    y = np.array([i for i in range(1, 8) for j in range(240)])
    X_train = feats[train]
    X_test = feats[test]
    y_train = y[train]
    y_test = y[test]
    return X_train, X_test, y_train, y_test


def get_img_names(ammount = 240):
    names_list = [('faces2/face_' + str(i).zfill(3) + '_' + str(j).zfill(5) + '.png')
                  for i in range(1, 8) for j in range(1, ammount + 1)]
    return names_list


def extraction_routine_LBP(arr_name, lbp_grids, lbp_dists):
    """ ONLY LBP
    Saves to arr_name .npy the array of features extracted.
    Params MUST be tuples, use (1,) for single values
    :param lbp_grids: tuple
    :param lbp_dists: tuple, must be the same size than ibp_grids
    :return: void
    """
    names = get_img_names()
    images = [cv2.imread(names[i], 0) for i in range(len(names))]

    lbps = []
    for i in range(len(lbp_grids)):
        print('Iteration{}/{}'.format((i + 1), len(lbp_grids)))
        with mp.Pool() as p:
            lbp = p.starmap(get_LBP, [(images[j], lbp_dists[i], lbp_grids[i]) for j in range(len(images))])
        lbps.append(lbp)

    lbp_feats = np.concatenate(lbps, axis=1)
    np.save(arr_name, lbp_feats)
    return


def extraction_routine_HAR(arr_name, lbp_grids, lbp_dists):
    """ ONLY HAR
    Saves to arr_name .npy the array of features extracted.
    Params MUST be tuples, use (1,) for single values
    :param lbp_grids: tuple
    :param lbp_dists: tuple, must be the same size than ibp_grids
    :return: void
    """
    names = get_img_names()
    images = [cv2.imread(names[i], 0) for i in range(len(names))]

    lbps = []
    for i in range(len(lbp_grids)):
        print('Iteration {}/{}'.format(str((i + 1)).zfill(2), str(len(lbp_grids)).zfill(2)))
        with mp.Pool() as p:
            lbp = p.starmap(get_Haralick, [(images[j], lbp_dists[i], lbp_grids[i]) for j in range(len(images))])
        lbps.append(lbp)

    lbp_feats = np.concatenate(lbps, axis=1)
    np.save(arr_name, lbp_feats)
    return


def extraction_routine_GAB_1(arr_name, gab_grids):
    """ ONLY HAR
    Saves to arr_name .npy the array of features extracted.
    Params MUST be tuples, use (1,) for single values
    :param lbp_grids: tuple
    :param lbp_dists: tuple, must be the same size than ibp_grids
    :return: void
    """
    names = get_img_names()
    images = [cv2.imread(names[i], 0) for i in range(len(names))]
    lbps = []
    for i in range(len(gab_grids)):
        print('Iteration {}/{}'.format(str((i + 1)).zfill(2), str(len(gab_grids)).zfill(2)))
        with mp.Pool() as p:
            lbp = p.starmap(get_Gab, [(images[j], gab_grids[i]) for j in range(len(images))])
        lbps.append(lbp)

    lbp_feats = np.concatenate(lbps, axis=1)
    np.save(arr_name, lbp_feats)
    return lbp_feats


def extraction_routine_GAB_2(arr_name, gab_grids):
    """ ONLY HAR
    Saves to arr_name .npy the array of features extracted.
    Params MUST be tuples, use (1,) for single values
    :param lbp_grids: tuple
    :param lbp_dists: tuple, must be the same size than ibp_grids
    :return: void
    """
    names = get_img_names()
    images = [cv2.imread(names[i], 0) for i in range(len(names))]
    lbps = []
    for i in range(len(gab_grids)):
        print('Iteration {}/{}'.format(str((i + 1)).zfill(2), str(len(gab_grids)).zfill(2)))
        with mp.Pool() as p:
            lbp = p.starmap(get_Gab_real_im, [(images[j], gab_grids[i]) for j in range(len(images))])
        lbps.append(lbp)

    lbp_feats = np.concatenate(lbps, axis=1)
    np.save(arr_name, lbp_feats)
    return lbp_feats


def feat_parameter_test_routine_1():
    """
    For LBP
    :return:
    """
    n = 10
    dist_tuple = tuple(range(1, 21))
    extraction_routine_LBP('feats', (n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n), dist_tuple)
    feats = np.load('feats.npy')
    print(len(feats[0]))
    for i in range(20):
        index = np.array(range(int((n ** 2) * (i * 59)), int((n ** 2) * ((i + 1) * 59))), np.dtype(int))
        if i == 0: print(len(index) * 2 * 10)
        matrix = feats[:, index]
        X_train, X_test, y_tr, y_te = separate_train_test(matrix)
        X_tr, X_te = dim_red_auto_PCA(X_train, X_test, .995)
        print(dist_tuple[i])
        k = classification_knn(X_tr, X_te, y_tr, y_te, 3)


def feat_parameter_test_routine_2():
    """
    For Haralick
    :return:
    """
    n = 2
    dist_tuple = tuple(range(1, 21))
    extraction_routine_HAR('feats', (n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n), dist_tuple)
    feats = np.load('feats.npy')
    print(len(feats[0]))
    for i in range(20):
        index = np.array(range(int((n ** 2) * (i * 52)), int((n ** 2) * ((i + 1) * 52))), np.dtype(int))
        if i == 0: print(len(index) * 2 * 10)
        matrix = feats[:, index]
        X_train, X_test, y_tr, y_te = separate_train_test(matrix)
        X_tr, X_te = dim_red_auto_PCA(X_train, X_test, .99)
        print(dist_tuple[i])
        k = classification_knn(X_tr, X_te, y_tr, y_te, 3)
    return


def feat_parameter_test_routine_3():
    """
        For normal gabor
        :return:
        """
    n = 2
    dist_tuple = (1, 2, 5, 10)
    extraction_routine_GAB_1('feats', dist_tuple)
    feats = np.load('feats.npy')
    print(len(feats[0]))
    for i in range(4):
        index = np.array(range(int((n ** 2) * (i * 96)), int((n ** 2) * ((i + 1) * 96))), np.dtype(int))
        if i == 0: print(len(index) * 2 * 10)
        matrix = feats[:, index]
        X_train, X_test, y_tr, y_te = separate_train_test(matrix)
        X_tr, X_te = dim_red_auto_PCA(X_train, X_test, .99)
        print(dist_tuple[i])
        k = classification_knn(X_tr, X_te, y_tr, y_te, 3)
    return


def feat_parameter_test_routine_4():
    """
        For real and imaginary gabor
        :return:
        """
    n = 2
    dist_tuple = (1, 2, 5, 10)
    extraction_routine_GAB_2('feats', dist_tuple)
    feats = np.load('feats.npy')
    print(len(feats[0]))
    for i in range(4):
        index = np.array(range(int((n ** 2) * (i * 192)), int((n ** 2) * ((i + 1) * 192))), np.dtype(int))
        if i == 0: print(len(index) * 2 * 10)
        matrix = feats[:, index]
        X_train, X_test, y_tr, y_te = separate_train_test(matrix)
        X_tr, X_te = dim_red_auto_PCA(X_train, X_test, .99)
        print(dist_tuple[i])
        k = classification_knn(X_tr, X_te, y_tr, y_te, 3)
    return
