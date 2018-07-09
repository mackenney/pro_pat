import librerias_patrones as lib_pat
import time
import sys
import pickle

import cv2

import classification

import os, glob

import multiprocessing as mp
import numpy as np

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, IncrementalPCA


class Image:
    def __init__(self, name, npy=False):
        self.group = int(name[-13:-10])
        self.number = int(name[-9:-4])


def get_standard_feats():
    path = 'image_features'
    names = glob.glob(os.path.join(path, '') + '*.npy')
    names = np.sort(np.array(names))
    # with mp.Pool() as p:
    #     actual_feats = p.map(np.load, [names[k] for k in range(len(names))])
    feats = []
    n = len(names)
    for i in range(n // 100 + 1):
        with mp.Pool() as p:
            f = p.map(np.load, [names[i * 100 + j] for j in range(100) if i * 100 + j < n])
        f = np.array(f).astype(np.float16)
        actual = np.concatenate(f)
        del f
        if i == 0:
            feats = actual
        else:
            feats = np.concatenate((feats, actual))
        del actual
        if i % 50 == 0:
            print('{}/{}'.format(i, n // 100))
    return feats, np.load('standard_labels.npy')

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
                [np.load("./image_features/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy")[0]])
            for i in range(len(feats_image)):
                while len(feats_image[i][0]) == 1:
                    feats_image[i] = feats_image[i][0]
            try:
                feat = np.concatenate(feats_image, axis=1)
                feats.append([feat])
            except ValueError:
                raise
    feats = np.concatenate(feats, axis=1)
    feats = feats[0]
    print("")
    return (feats)

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


def batch_processing(feats, labels, pca_ratio, kpca_ratio, kpca_kernel, tsdv_ratio):
    new_feats = []
    picked = []
    counts = []
    T = int(np.max(labels) + 1)
    for i in range(T):
        print('{}/{}'.format(i, T))
        actual_feats = feats[:, np.nonzero(labels == i)[0]]
        pca = PCA()
        pca.fit(actual_feats)
        exp = 0.
        count = 0
        for i in range(len(actual_feats)):
            exp += pca.explained_variance_ratio_[i]
            if exp >= pca_ratio:
                count = i + 1
                break
        pca_feats = np.asarray(pca.transform(actual_feats))[:, 0:count]
        pca_instance = pickle.dumps(pca)
        pca_count = count

        kpca = KernelPCA(kernel=kpca_kernel, n_jobs=1)
        kpca_feats = kpca.fit_transform(actual_feats)
        t = np.sum(kpca.lambdas_)
        aux = 0
        index = 0
        for i in range(len(kpca.lambdas_)):
            if aux / t >= kpca_ratio:
                index = i
                break
            aux += kpca.lambdas_[i]
            index = i
        kpca_feats = kpca_feats[:, :index]
        kpca_instance = pickle.dumps(kpca)
        kpca_count = index

        tsdv = TruncatedSVD(n_components=len(actual_feats[0]) - 1)
        tsdv.fit(actual_feats)
        exp = 0.
        count = 0
        for i in range(len(actual_feats)):
            exp += tsdv.explained_variance_ratio_[i]
            if exp >= tsdv_ratio:
                count = i + 1
                break

        tsdv = TruncatedSVD(n_components=count)
        tsdv_feats = tsdv.fit_transform(actual_feats)
        tsdv_instance = pickle.dumps(tsdv)
        tsdv_count = count

        instances = (pca_instance, kpca_instance, tsdv_instance)
        f = np.concatenate((pca_feats, kpca_feats, tsdv_feats), axis=1)
        c = (pca_count, kpca_count, tsdv_count)
        counts.append(c)
        new_feats.append(f)
        picked.append(instances)
    final_feats = np.concatenate(new_feats, axis=1)
    return final_feats, picked, counts


def process_new_feats(f, labels, picked, counts):
    new_feats = []
    T = int(np.max(labels) + 1)
    for i in range(T):
        print('{}/{}'.format(i, T))
        actual_feats = f[:, np.nonzero(labels == i)[0]]

        pca = pickle.loads(picked[i][0])
        pca_feats = pca.transform(actual_feats)[:, counts[i][0]]

        kpca = pickle.loads(picked[i][1])
        kpca_feats = kpca.fit_transform(actual_feats)[:, counts[i][1]]

        tsdv = pickle.loads(picked[i][2])
        tsdv_feats = tsdv.transform(actual_feats)[:, counts[i][2]]

        f = np.concatenate((pca_feats, kpca_feats, tsdv_feats), axis=1)

        new_feats.append(f)
    final_feats = np.concatenate(new_feats, axis=1)
    return final_feats


def fast_batch_processing(feats, labels, pca_ratio):
    new_feats = []
    picked = []
    counts = []
    T = int(np.max(labels) + 1)
    for i in range(T):
        print('{}/{}'.format(i, T))

        hh = np.nonzero(labels == i)[0]
        if len(hh) == 0:
            continue

        actual_feats = feats[:, hh]


        pca = IncrementalPCA()
        try:
            pca_feats = pca.fit_transform(actual_feats)
        except:
            pca_feats = pca.fit_transform(
                actual_feats + 0.01 * np.random.randn(len(np.ravel(actual_feats))).reshape(actual_feats.shape))
        exp = 0.
        count = 0
        for i in range(len(actual_feats)):
            exp += pca.explained_variance_ratio_[i]
            if exp >= pca_ratio:
                count = i + 1
                break
        pca_feats = np.asarray(pca_feats)[:, :count]
        pca_instance = pickle.dumps(pca)
        pca_count = count

        f = pca_feats
        c = counts
        counts.append(c)
        new_feats.append(f)
        picked.append(pca_instance)
    final_feats = np.concatenate(new_feats, axis=1)
    return final_feats, picked, counts


def fast_process_new_feats(input, labels, picked, counts):
    new_feats = []
    T = int(np.max(labels) + 1)
    for i in range(T):
        print('{}/{}'.format(i, T))

        hh = np.nonzero(labels == i)[0]
        if len(hh) == 0:
            continue

        actual_feats = input[:, hh]

        pca = pickle.loads(picked[i])
        pca_feats = pca.transform(actual_feats)[:, counts[i]]

        f = pca_feats
        new_feats.append(f)
    final_feats = np.concatenate(new_feats, axis=1)
    return final_feats


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


def extract_test(path):
    files = glob.glob(path)
    files = np.sort(files)
    lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
    har_params = ((1, 1, 1, 2, 5), (1, 10, 20, 11, 8))
    gab1_params = (1, 2, 5, 10)
    gab2_params = (1, 2, 5, 10)

    # count = 0
    # bar_len = 60
    # total = len(files)
    if not os.path.isdir('./image_features_test'):
        os.mkdir('./image_features_test')
    # for name in files:
    #     img = Image(name)
    #     image = cv2.imread(name, 0)
    #     image = cv2.resize(image, (200, 200))
    #     lib_pat.progress(count, total, name)
    #     extraction_routine("./image_features_test/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
    #                        [image], lbp_params[0], lbp_params[1], har_params[0], har_params[1], gab1_params,
    #                        gab2_params)
    #     count += 1

    images = []
    names = []
    for name in files:
        img = Image(name)
        image = cv2.imread(name, 0)
        images.append(image)
        names.append("./image_features_test/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5))
        # lib_pat.progress(count, total, name)
    print('hola :)')
    extraction_routine(names, images, lbp_params[0], lbp_params[1], har_params[0], har_params[1], gab1_params,
                       gab2_params)

    print("")


def get_feats_test():
    path = './image_features_test/*.npy'
    files = glob.glob(path)
    feats = []
    count = 0
    bar_len = 60
    total = len(files)
    for name in files:
        count += 1
        lib_pat.progress(count, total, name)
        feats_image = []
        feats_image.append([np.load(name)[0]])
        for i in range(len(feats_image)):
            while len(feats_image[i][0]) == 1:
                feats_image[i] = feats_image[i][0]
        try:
            feat = np.concatenate(feats_image, axis=1)
            feats.append([feat])
        except ValueError:
            raise
    feats = np.concatenate(feats, axis=1)
    feats = feats[0]
    print("")
    return (feats)


def test(path):
    extract_test(path)
    feats = get_feats_test()


def clases_test(feats, path):
    files = glob.glob(path)
    y_test = []
    for name in files:
        img = Image(name)
        y_test.append(img.group)
    X_test = feats
    y_test = np.array(y_test)
    return X_test, y_test


if __name__ == '__main__':
    # print('standard feats')
    # f1, l1 = get_standard_feats()
    # print('landmark feats')
    # f2, l2 = get_landmark_feats()
    #
    # np.save('f1',f1)
    # np.save('f2',f2)
    # np.save('l1',l1)
    # np.save('l2',l2)
    # quit()
    # print('loading')
    # f1, f2, l1, l2 = np.load('f1.npy').astype(np.float16), np.load('f2.npy').astype(np.float16), \
    #                  np.load('l1.npy').astype(np.float16), np.load('l2.npy').astype(np.float16)
    # print('concatenating')
    # master_f = np.concatenate((f1, f2), axis=1)
    # m = np.max(l1) + 1
    # l2 += m
    # master_l = np.concatenate((l1, l2))
    # print('saving')
    # np.save('master_feats', master_f)
    # np.save('master_label', master_l)

    # mf = np.load('f1.npy')
    # ml = np.load('l1.npy')

    print(1)
    mf = get_feats(240)
    ml = np.load('l1.npy')

    rem_var_index = lib_pat.delete_zero_variance_features2(mf,ml,.1)

    mf, ml = mf[:, rem_var_index], ml[rem_var_index]
    print(2)
    feats, pickled, counts = fast_batch_processing(mf, ml, .6)

    X_tr, X_te, y_tr,y_te = lib_pat.hold_out(mf, 240)
    print(3)
    #entrenar
    lda = classification.fitted_classifier_LDA(X_tr, y_tr)
    print(31)
    mlp = classification.fitted_classifier_MLP(X_tr, y_tr)
    print(4)
    np.save('lda_classifier', lda)
    np.save('mlp_classifier', mlp)

    # guardar lda, mlp

    np.save('pcas', pickled)
    pickled = np.load('pcas.npy')

    print(5)
    extract_test('./test/*.png')
    test_feats = get_feats_test()

    test_feats = test_feats[rem_var_index]

    test_feats = fast_process_new_feats(test_feats,ml,pickled,counts)

    y_test = [i for i in range(7) for j in range(4)]

    ff = fast_process_new_feats(test_feats, ml, pickled, counts)

    classification.classify(lda,ff,y_test)
