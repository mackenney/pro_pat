# import the necessary packages
from imutils import face_utils
import numpy as np
import dlib
import cv2
import glob
from extract_features import *
import os
import librerias_patrones as lib_pat


def landmark_ext_routine(img_path, num_index, threads=False):
    """
    Saves in folder the facial landmarks for each image.
    :param img_path: image source
    :param dest_path: array destination
    :param num_index: index of pictures to look at.
    :return: void
    """
    if img_path is None:
        path = './faces/*.png'
    else:
        path = img_path
    files = glob.glob(path)

    if not os.path.isdir('./landmarks'):
        os.mkdir('./landmarks')

    if threads:
        new_names = []
        for name in files:
            group = int(name[-13:-10])
            number = int(name[-9:-4])
            if (number in num_index):
                new_names.append(name)
        import multiprocessing as mp
        with mp.Pool() as p:
            p.map(_save_landmarks, new_names)
    else:
        for name in files:
            group = int(name[-13:-10])
            number = int(name[-9:-4])
            if (number in num_index):
                landmarks = _get_landmarks(name, False)[0]
                np.save('./landmarks/face_{}_{}.png'.format(str(group).zfill(3), str(number).zfill(5)), landmarks)


def crop_landmark(image, landmarks, part, slack=0., show_crop=False):
    """
    Returns an image from a selected landmark
    part =
    0 left eyebrow
    1 right eyebrow
    2 nose
    3 left eye
    4 right eye
    5 mouth

    :param image:
    :param landmarks:
    :param part:
    :param slack:
    :return:
    """
    if (part == "left eyebrow" or part == 0):
        rango = range(17, 22)
    elif (part == "right eyebrow" or part == 1):
        rango = range(22, 27)
    elif (part == "nose" or part == 2):
        rango = range(27, 36)
    elif (part == "left eye" or part == 3):
        rango = range(36, 42)
    elif (part == "right eye" or part == 4):
        rango = range(42, 48)
    elif (part == "mouth" or part == 5):
        rango = range(48, 68)

    landmarks = np.array(landmarks)
    rango = np.array(rango)
    x_max = landmarks[rango, 0].max()
    x_min = landmarks[rango, 0].min()
    y_max = landmarks[rango, 1].max()
    y_min = landmarks[rango, 1].min()

    x_slack = int(np.ceil((x_max - x_min + 1) * slack))
    y_slack = int(np.ceil((y_max - y_min + 1) * slack))

    if x_max - x_min < 2:
        x_slack += 1

    landmark = image[max(y_min - y_slack, 0):y_max + y_slack, max(x_min - x_slack, 0):x_max[0] + x_slack]
    if show_crop:
        cv2.imshow("Image", landmark)
        cv2.waitKey(15000)
        # cv2.waitKey(0)
    return landmark


def extract_landmarks_feats_with_threads(index, feature, overwrite=False):
    """
    extract a feature from landmarks crops from images with matching index using threads and saves them automatically
    :param index: numbers to consider
    :param feature: 0,1,2 = lbp, har, tas
    :return:
    """
    path = './faces/*.png'
    files = glob.glob(path)
    params = (1, 5, 8)

    if feature == 0:
        feat = 'lbp'
    elif feature == 1:
        feat = 'har'
    else:
        feat = 'tas'

    if not os.path.isdir('./eyebrowL/{}'.format(feat)):
        os.mkdir('./eyebrowL/{}'.format(feat))
    if not os.path.isdir('./eyebrowR/{}'.format(feat)):
        os.mkdir('./eyebrowR/{}'.format(feat))
    if not os.path.isdir('./nose/{}'.format(feat)):
        os.mkdir('./nose/{}'.format(feat))
    if not os.path.isdir('./eyeL/{}'.format(feat)):
        os.mkdir('./eyeL/{}'.format(feat))
    if not os.path.isdir('./eyeR/{}'.format(feat)):
        os.mkdir('./eyeR/{}'.format(feat))
    if not os.path.isdir('./mouth/{}'.format(feat)):
        os.mkdir('./mouth/{}'.format(feat))

    images = []
    names = []
    landmarks_points = []

    feat_path_prefix = "./{}/{}/face_".format('{}', feat)

    print('Fetching images and landmark points...')
    for name in files:
        img = Image(name)
        feat_path = feat_path_prefix + str(img.group).zfill(3) + "_" + str(img.number).zfill(5)
        if img.number in index and (overwrite or (not os.path.isfile(feat_path.format('eyebrowL') + '.npy'))):
            try:
                names.append(img)
                images.append(cv2.imread(name, 0))
                landmarks_points.append(np.load(
                    "./landmarks/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".png.npy"))
            except:
                print('Error in: {}'.format(name))

    print('Cropping images...')
    with mp.Pool() as p:
        lb_crops = p.starmap(crop_landmark,
                             [(images[k], landmarks_points[k], 'left eyebrow', 0.15) for k in range(len(images))])
        rb_crops = p.starmap(crop_landmark,
                             [(images[k], landmarks_points[k], 'right eyebrow', 0.15) for k in range(len(images))])
        no_crops = p.starmap(crop_landmark,
                             [(images[k], landmarks_points[k], 'nose', 0.15) for k in range(len(images))])
        le_crops = p.starmap(crop_landmark,
                             [(images[k], landmarks_points[k], 'left eye', 0.15) for k in range(len(images))])
        re_crops = p.starmap(crop_landmark,
                             [(images[k], landmarks_points[k], 'left eye', 0.15) for k in range(len(images))])
        mo_crops = p.starmap(crop_landmark,
                             [(images[k], landmarks_points[k], 'mouth', 0.15) for k in range(len(images))])

        if feat == 'har':
            # check that the distance params are not greater than the min dimension of the crops.
            print('checking landmark crops dimensions...')
            shapes = []
            for i in range(len(lb_crops)):
                shapes.append(lb_crops[i].shape)
                shapes.append(rb_crops[i].shape)
                shapes.append(no_crops[i].shape)
                shapes.append(le_crops[i].shape)
                shapes.append(rb_crops[i].shape)
                shapes.append(mo_crops[i].shape)
            shapes = np.array(shapes)
            min_shapes = np.array(shapes).min(axis=0)

            x_arg_min = np.argmin(shapes[:, 0])
            y_arg_min = np.argmin(shapes[:, 1])
            ind = x_arg_min // 6
            print('min: {}'.format(min_shapes))
            par = []
            for i in params:
                if i <= min_shapes[0] and i <= min_shapes[1]:
                    par.append(i)
            params = tuple(par)

        print('extracting features...')
        lb_features = p.starmap(_ext, [(lb_crops[k], params, feature) for k in range(len(lb_crops))])
        print('1/6')
        rb_features = p.starmap(_ext, [(rb_crops[k], params, feature) for k in range(len(lb_crops))])
        print('2/6')
        no_features = p.starmap(_ext, [(no_crops[k], params, feature) for k in range(len(lb_crops))])
        print('3/6')
        le_features = p.starmap(_ext, [(le_crops[k], params, feature) for k in range(len(lb_crops))])
        print('4/6')
        re_features = p.starmap(_ext, [(re_crops[k], params, feature) for k in range(len(lb_crops))])
        print('5/6')
        mo_features = p.starmap(_ext, [(mo_crops[k], params, feature) for k in range(len(lb_crops))])
        print('6/6')

    print('saving features...')
    n = len(lb_crops)
    for k in range(n):
        if k % 50 == 0:
            print('{}/{}'.format(k, n))
        feat_path = feat_path_prefix + str(names[k].group).zfill(3) + "_" + str(names[k].number).zfill(5)
        np.save(feat_path.format('eyebrowL'), lb_features[k])
        np.save(feat_path.format('eyebrowR'), rb_features[k])
        np.save(feat_path.format('nose'), no_features[k])
        np.save(feat_path.format('eyeL'), le_features[k])
        np.save(feat_path.format('eyeR'), re_features[k])
        np.save(feat_path.format('mouth'), mo_features[k])


def show_landmarks():
    """
    Shows the images with the landmark points marked
    :param num_index:
    :return:
    """
    path = './landmarks/*.npy'
    files = glob.glob(path)
    for name in files:
        lm = np.load(name)
        img_name = name.replace('landmarks', 'faces').replace('.npy', '')
        image = cv2.imread(img_name)
        for i in range(len(lm)):
            try:
                # image[lm[i][1]][lm[i][0]] = 0
                cv2.circle(image, (lm[i][0], lm[i][1]), 1, (0, 0, 255), -1)
            except IndexError:
                pass
        cv2.imshow('Image', image)
        cv2.waitKey(5000)


# Deprecated and hidden for internal use. #


def _ext(image, dists, feat):
    """
    helper function
    0: lbp
    1: har
    2: tas
    :param image:
    :param dists:
    :param feat:
    :return:
    """
    f = []
    for d in dists:
        if feat == 0:
            f.append(lib_pat.get_LBP(image, d))
        elif feat == 1:
            f.append(lib_pat.get_Haralick(image, d))
        else:
            f.append(lib_pat.get_TAS(image, 1))
    if len(f) == 1:
        return np.array(f[0])
    else:
        return np.concatenate(f)


def _get_landmarks(input, show_image=False):
    """
    Takes an image and returns an array of facial landmarks and boundbox (x, y, w, h)
    :param input:
    :return:
    """
    if type(input) == str:
        im = cv2.imread(input)
        if im.shape[2] == 3:
            image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            image = im
    elif isinstance(input, np.ndarray):
        im = input
        if im.shape[2] == 3:
            image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            image = im
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    gray = image
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) == 0:
        # foto entera es el rectangulo
        rectangle = None
        rectangle = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
    elif len(rects) == 1:
        # ok
        rectangle = rects[0]
    else:
        # Ahora se elige el más grande.
        sizes = []
        for r in rects:
            (x, y, w, h) = face_utils.rect_to_bb(r)
            sizes.append(w * h)
        rectangle = rects[np.argmax(sizes)]
    rect = rectangle
    # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a
    # NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    if show_image:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(image, "Face", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    return shape, x, y, w, h


def _save_landmarks(name):
    """
    From a file name calls _get_landmarks and saves the array.
    :param name:
    :return:
    """
    group = int(name[-13:-10])
    number = int(name[-9:-4])
    landmarks = _get_landmarks(name, False)[0]
    np.save('./landmarks/face_{}_{}.png'.format(str(group).zfill(3), str(number).zfill(5)), landmarks)


if __name__ == '__main__':
    # landmarks, x, y, w, h = _get_landmarks('me1.jpg', True)
    # im = cv2.imread('me1.jpg')
    # crop_landmark(im, landmarks, 0, 0.1, True)

    import time

    tt = time.time()
    print('extracting landmarks...')
    landmark_ext_routine(None, np.arange(0, 2013), threads=True)  # care! must include last one!
    print('time taken:', time.time() - tt)
    # quit()
    # show_landmarks(np.arange(150))

    tt = time.time()
    print('begining lbp...')
    extract_landmarks_feats_with_threads(np.arange(2015), 0, overwrite=True)
    print('time taken:', time.time() - tt)

    tt = time.time()
    print('begining har...')
    extract_landmarks_feats_with_threads(np.arange(2015), 1, overwrite=True)
    print('time taken:', time.time() - tt)

    tt = time.time()
    print('begining tas...')
    extract_landmarks_feats_with_threads(np.arange(2015), 2, overwrite=True)
    print('time taken:', time.time() - tt)
