# import the necessary packages
from imutils import face_utils
import numpy as np
import dlib
import cv2
import glob
from extract_features import *
import os

def get_landmarks(input, show_image=False):
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


def save_landmarks(name):
    group = int(name[-13:-10])
    number = int(name[-9:-4])
    landmarks = get_landmarks(name, False)[0]
    np.save('./landmarks/face_{}_{}.png'.format(str(group).zfill(3), str(number).zfill(5)), landmarks)



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
            p.map(save_landmarks, new_names)
    else:
        for name in files:
            group = int(name[-13:-10])
            number = int(name[-9:-4])
            if (number in num_index):
                landmarks = get_landmarks(name, False)[0]
                np.save('./landmarks/face_{}_{}.png'.format(str(group).zfill(3),str(number).zfill(5)), landmarks)




def crop_landmark(image, landmarks, part, slack=0, show_crop=False):
    """
    Returns an image from a selected landmark
    part =
    0 left ceja
    1 right ceja
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

    x_min = [10000, 10000]
    x_max = [0, 0]
    y_min = [10000, 10000]
    y_max = [0, 0]
    landmark = image.copy()

    for i in rango:
        landmark[landmarks[i][1]][landmarks[i][0]] = 0
        if (landmarks[i][0] > x_max[0]):
            x_max = landmarks[i]
        if (landmarks[i][0] < x_min[0]):
            x_min = landmarks[i]
        if (landmarks[i][1] > y_max[1]):
            y_max = landmarks[i]
        if (landmarks[i][1] < y_min[1]):
            y_min = landmarks[i]
    x_slack = int((x_max[0] - x_min[0]) * slack)
    y_slack = int((y_max[1] - y_min[1]) * slack)
    landmark = image[y_min[1] - y_slack:y_max[1] + y_slack, x_min[0] - x_slack:x_max[0] + x_slack]
    if show_crop:
        cv2.imshow("Image", landmark)
        cv2.waitKey(15000)
        # cv2.waitKey(0)
    return landmark


### needs refactoring
def extract_landmarks(start, end):
    path = './faces/*.png'
    files = glob.glob(path)
    number_of_features = end - start + 1

    lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
    har_params = ()
    gab1_params = ()
    gab2_params = ()

    count = 0
    bar_len = 60
    total = number_of_features * 7
    for name in files:
        img = Image(name)
        if (img.number >= start and img.number <= end):
            image = cv2.imread(name, 0)
            landmarks = np.load("./landmarks/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy")
            lib_pat.progress(count, total, name)
            ### LEFT EYEBROW
            landmark = crop_landmark(image, landmarks, "left eyebrow", slack=0.15)
            if not os.path.isdir('./eyebrowL'):
                os.mkdir('./eyebrowL')
            extraction_routine("./eyebrowL/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params,
                               gab2_params)
            ### RIGHT EYEBROW
            landmark = crop_landmark(image, landmarks, "right eyebrow", slack=0.15)
            if not os.path.isdir('./eyebrowR'):
                os.mkdir('./eyebrowR')
            extraction_routine("./eyebrowR/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params,
                               gab2_params)
            ### NOSE
            landmark = crop_landmark(image, landmarks, "nose", slack=0.1)
            if not os.path.isdir('./nose'):
                os.mkdir('./nose')
            extraction_routine("./nose/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params,
                               gab2_params)
            ### LEFT EYE
            landmark = crop_landmark(image, landmarks, "left eye", slack=0.1)
            if not os.path.isdir('./eyeL'):
                os.mkdir('./eyeL')
            extraction_routine("./eyeL/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params,
                               gab2_params)
            ### RIGHT EYE
            landmark = crop_landmark(image, landmarks, "right eye", slack=0.1)
            if not os.path.isdir('./eyeR'):
                os.mkdir('./eyeR')
            extraction_routine("./eyeR/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params,
                               gab2_params)
            ### MOUTH
            landmark = crop_landmark(image, landmarks, "mouth", slack=0.1)
            if not os.path.isdir('./mouth'):
                os.mkdir('./mouth')
            extraction_routine("./mouth/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5),
                               [landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params,
                               gab2_params)
            count += 1
    print("")


if __name__ == '__main__':
    # landmarks, x, y, w, h = get_landmarks('me1.jpg', True)
    # im = cv2.imread('me1.jpg')
    # crop_landmark(im, landmarks, 0, 0.1, True)


    import time
    tt = time.time()
    landmark_ext_routine(None, np.arange(250, 1000), True)
    print(time.time()-tt)