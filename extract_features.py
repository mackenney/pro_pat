import librerias_patrones as lib_pat
import classification
import main
import cv2
import glob
import os
import numpy as np
import multiprocessing as mp
from sklearn.metrics import confusion_matrix as CM


class Image:
    def __init__(self, name, npy=False):
        self.group = int(name[-13:-10])
        self.number = int(name[-9:-4])
    # if npy:
    # 	print(name[-13:-10])
    # 	print(name[-9:-4])
    # 	self.group = int(name[-13:-10])
    # 	self.number = int(name[-9:-4])
    # else:
    # 	self.group = int(name[14:17])
    # 	self.number = int(name[18:23])


def extraction_routine(arr_name, images, lbp_grids, lbp_dists, har_grids, har_dists, gab_grids1, gab_grids2):
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
    count = 0
    total = len(lbp_grids) + len(har_grids) + len(gab_grids1) + len(gab_grids2)
    lbps = []
    for i in range(len(lbp_grids)):
        # print('LPB Extraction, Iteration {}/{}'.format((i + 1), len(lbp_grids)))
        with mp.Pool() as p:
            lbp = p.starmap(lib_pat.get_LBP, [(images[j], lbp_dists[i], lbp_grids[i], j) for j in range(len(images))])
        lbps.append(lbp)
        count += 1
        lib_pat.progress(count, total, up=False)
    lbp_feats = np.concatenate(lbps, axis=1)

    hars = []
    for i in range(len(har_grids)):
        # print('Haralick Extraction, Iteration {}/{}'.format((i + 1), len(har_grids)))
        with mp.Pool() as p:
            har = p.starmap(lib_pat.get_Haralick,
                            [(images[j], har_dists[i], har_grids[i], j) for j in range(len(images))])
        hars.append(har)
        count += 1
        lib_pat.progress(count, total, up=False)
    har_feats = np.concatenate(hars, axis=1)

    gabs1 = []
    for i in range(len(gab_grids1)):
        # print('Gabor Extraction, Iteration {}/{}'.format((i + 1), len(gab_grids1)))
        with mp.Pool() as p:
            gab = p.starmap(lib_pat.get_Gab, [(images[j], gab_grids1[i], j) for j in range(len(images))])
        gabs1.append(gab)
        count += 1
        lib_pat.progress(count, total, up=False)
    gab_feats1 = np.concatenate(gabs1, axis=1)

    gabs2 = []
    for i in range(len(gab_grids2)):
        # print('Gabor Extraction, Iteration {}/{}'.format((i + 1), len(gab_grids2)))
        with mp.Pool() as p:
            gab = p.starmap(lib_pat.get_Gab_real_im, [(images[j], gab_grids2[i], j) for j in range(len(images))])
        gabs2.append(gab)
        count += 1
        lib_pat.progress(count, total, up=False)
    gab_feats2 = np.concatenate(gabs2, axis=1)

    feats = np.concatenate((lbp_feats, har_feats, gab_feats1, gab_feats2), axis=1)
    np.save(arr_name, feats)
    return feats


def classify(feats, cantidad):
    lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
    har_params = ((1, 1, 1, 2, 5), (1, 10, 20, 11, 8))
    gab1_params = (1, 2, 5, 10)
    gab2_params = (1, 2, 5, 10)
    params_landmarks = (1, 5, 8)
    labels_image = main.generate_labels(lbp_params[0], har_params[0], gab1_params, gab2_params)
    # print(labels_image)
    # print(len(labels_image))
    labels_landmarks = main.generate_labels_landmarks(labels_image[-1] + 1, 6, params_landmarks, (), (1))
    # print(labels_landmarks)
    # print(len(labels_landmarks))
    labels = np.concatenate([labels_image, labels_landmarks], axis=0)
    # print(labels)
    # print(len(labels))

    # labels = labels_image

    print(labels)

    print('Removing features with low variance')
    rem_var_index = lib_pat.delete_zero_variance_features2(feats, labels, 0.1)
    np.save('rem_var_index.npy', rem_var_index)
    feats, labels = feats[:, rem_var_index], labels[:, rem_var_index]

    print('Separating Features...')
    X_tr, X_te, y_tr, y_te = lib_pat.hold_out(feats, cantidad)

    print('Reducing features by transformation')
    X_tr, X_te = main.reduction_routine(feats, labels, .99, cantidad)
    print('Final reduction (for no colinear features)')
    X_tr, X_te = lib_pat.dim_red_auto_PCA(X_tr, X_te, ratio=.9)

    np.save("X_tr_" + str(cantidad), X_tr)
    np.save("X_te_" + str(cantidad), X_te)
    np.save("y_tr_" + str(cantidad), y_tr)
    np.save("y_te_" + str(cantidad), y_te)

    print('Classification via LDA solver=svd')
    k1, k1_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te, solver='svd')

    print('Classification via MLP')
    k2, k2_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te)

    print('Classification via NN')
    k3, k3_score = classification.training_and_classification_NN(X_tr, X_te, y_tr, y_te)

    np.savetxt('k1', CM(y_te, k1), fmt='%2i', delimiter=',')
    np.savetxt('k2', CM(y_te, k2), fmt='%2i', delimiter=',')
    np.savetxt('k3', CM(y_te, k3), fmt='%2i', delimiter=',')


def extract(start, end):
    path = './faces/*.png'
    files = glob.glob(path)
    number_of_features = end - start + 1

    lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
    har_params = ((1, 1, 1, 2, 5), (1, 10, 20, 11, 8))
    gab1_params = (1, 2, 5, 10)
    gab2_params = (1, 2, 5, 10)

    count = 0
    bar_len = 60
    total = number_of_features * 7
    for name in files:
        img = Image(name)
        if (img.number >= start and img.number <= end):
            image = cv2.imread(name, 0)
            lib_pat.progress(count, total, name)
            extraction_routine("./image_features/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [image], lbp_params[0], lbp_params[1], har_params[0], har_params[1], gab1_params,
                               gab2_params)
            count += 1
        # filled_len = int(round(bar_len * count / float(total)))

        # percents = round(100.0 * count / float(total), 1)
        # bar = '=' * filled_len + '-' * (bar_len - filled_len)

        # sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', name))
        # sys.stdout.flush()
    print("")


def extract2(start, end):
    path = './proyecto_features/*.npy'
    files = glob.glob(path)
    for name in files:
        if (name == "./proyecto_features/very_huge_raw_features.npy"):
            feats = np.load(name)

    for i in range(len(feats)):
        number = (i % 240) + 1
        group = (i // 240) + 1
        arr_name = "./image_features2/face_" + str(group).zfill(3) + "_" + str(number).zfill(5) + ".npy"
        np.save(arr_name, np.array([feats[i]]))


def get_feats(number_of_features):
    path = './image_features/*.npy'
    files = glob.glob(path)
    feats = []
    count = 0
    bar_len = 60
    total = number_of_features * 7
    for name in files:
        # print(name)
        img = Image(name, npy=True)
        if (img.number <= number_of_features):
            count += 1
            lib_pat.progress(count, total, name)
            feats_image = []
            feats_image.append(
                np.load("./image_features/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyebrowL/lbp/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyebrowR/lbp/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./nose/lbp/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyeL/lbp/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyeR/lbp/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./mouth/lbp/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyebrowL/tas/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyebrowR/tas/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./nose/tas/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyeL/tas/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./eyeR/tas/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
            feats_image.append(
                np.load("./mouth/tas/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))

        feat = np.concatenate(feats_image, axis=1)
        feats.append([feat[0]])
        feats = np.concatenate(feats, axis=1)
        feats = feats[0]
    print("")
    return (feats)


def landmark_classifier(feats, cantidad, iterations, separate_ratio):
    params_landmarks = (1, 5, 8)
    labels_landmarks = main.generate_labels_landmarks(0, 1, params_landmarks, (), (1))
    print(labels_landmarks)

    print('Removing features with low variance')
    feats, labels = lib_pat.delete_zero_variance_features(feats, labels_landmarks, 0.05)

    lda_scores = []
    mlp_scores = []
    for i in range(iterations):
        print('Classification Nº {}/{}'.format((i + 1), iterations))
        print('Separating Features...')
        X_tr, X_te, y_tr, y_te, sep_list = lib_pat.separate_train_test(feats, separate_ratio, cantidad)

        print('Reducing features by transformation')
        X_tr, X_te = main.reduction_routine(feats, labels, separate_ratio, .99, cantidad, sep_list)
        print('Final reduction (for no colinear features)')
        X_tr, X_te = lib_pat.dim_red_auto_PCA(X_tr, X_te, ratio=.9)

        print('Classification via LDA solver=svd')
        k1, k1_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te, solver='svd')
        lda_scores.append(k1_score)

        print('Classification via MLP')
        k2, k2_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te)
        mlp_scores.append(k2_score)

        np.savetxt('k1', CM(y_te, k1), fmt='%2i', delimiter=',')
        np.savetxt('k2', CM(y_te, k2), fmt='%2i', delimiter=',')
    lda_mean = sum(lda_scores) / float(len(lda_scores))
    print('LDA mean accuracy:', lda_mean)
    mlp_mean = sum(mlp_scores) / float(len(mlp_scores))
    print('MLP mean accuracy:', mlp_mean)


def classify_trained(cantidad):
    X_tr = np.load("X_tr_" + str(cantidad) + ".npy")
    X_te = np.load("X_te_" + str(cantidad) + ".npy")
    y_tr = np.load("y_tr_" + str(cantidad) + ".npy")
    y_te = np.load("y_te_" + str(cantidad) + ".npy")

    print('Classification via LDA solver=svd')
    k1, k1_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te, solver='svd')

    print('Classification via MLP')
    k2, k2_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te)

    print('Classification via NN')
    k3, k3_score = classification.training_and_classification_NN(X_tr, X_te, y_tr, y_te)

    np.savetxt('k1', CM(y_te, k1), fmt='%2i', delimiter=',')
    np.savetxt('k2', CM(y_te, k2), fmt='%2i', delimiter=',')
    np.savetxt('k3', CM(y_te, k3), fmt='%2i', delimiter=',')


def get_landmark_feats():
    landmark_names = ['eyebrowL', 'eyebrowR', 'eyeL', 'eyeR', 'mouth', 'nose']
    feats = []
    labs = []
    for lm in landmark_names:
        print('now in:', lm)
        actual_feats, actual_labs = _load_feats_from_folder(lm)
        feats.append(actual_feats)
        if len(labs) == 0:
            m = 0
        else:
            m = np.max(labs) + 1
        labs.append(actual_labs + m)
    feats = np.concatenate(feats, axis=1)
    labs = np.concatenate(labs)
    return feats, labs


def get_standard_feats():
    path = 'image_features'
    names = glob.glob(os.path.join(path, '') + '*.npy')
    names = np.sort(np.array(names))
    with mp.Pool() as p:
        actual_feats = p.map(np.load, [names[k] for k in range(len(names))])
    f = np.stack(actual_feats)
    return f, np.load('standard_labels.npy')


def _load_feats_from_folder(path):
    """
    Returns all the features inside the subfolders of the given path and their labels.
    :param path:
    :return:
    """
    folders = []
    if os.path.isdir(path):
        folders = []
        for i in os.scandir(path):
            if os.path.isdir(i):
                folders.append(i.path)
    feats = []
    labs = []
    counter = 0
    for f in folders:
        names = glob.glob(os.path.join(f, '') + '*.npy')
        names = np.sort(np.array(names))
        with mp.Pool() as p:
            actual_feats = p.map(np.load, [names[k] for k in range(len(names))])
        feats.append(actual_feats)
        labs.append(counter * np.ones(len(actual_feats[0])))
        counter += 1

    if len(feats) == 0:
        print('there is no features to extract!!')
        print(path)
        raise ValueError
    elif len(feats) == 1:
        return np.array(feats), np.array(labs)
    else:
        return np.concatenate(feats, axis=1), np.concatenate(labs)


if __name__ == '__main__':
    # start = 1901
    # end = 2012
    end = 240
    # # extract(start, end)
    # feats = get_feats(end)
    # classify(feats, end)
    # classify_trained(end)

    f, l = get_standard_feats()
    np.save('standard_feats', f)
