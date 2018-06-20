import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np


def detect_faces(image):
    """
    Obtains the rectangles that mark the faces in an image
    :param image: np array
    :return: list of tuples. Each tuple represents a rectangle
    """

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames


def generate_cropped_nparray(img_path):
    """
    Gets the matrix (numpy array) of the cropped face
    :param img_path:
    :return: np array
    """
    image = io.imread(img_path)

    detected_faces = detect_faces(image)
    if len(detected_faces) != 1:
        print('Imagen {} genera {} caras'.format(img_path, len(detected_faces)))
    else:
        pil = Image.fromarray(image).crop(detected_faces[0])
        return np.asarray(pil)


def save_cropped_image(img_path, crop_path):
    """
    Saves the cropped face as an independent image
    :param img_path:
    :param crop_path:
    """
    image = io.imread(img_path)

    detected_faces = detect_faces(image)
    if len(detected_faces) != 1:
        print('Imagen {} genera {} caras'.format(img_path, len(detected_faces)))
    else:
        face = Image.fromarray(image).crop(detected_faces[0])
        io.imsave(crop_path, face)


if __name__ == "__main__":
    # Load image
    #img_path = './CFD 2.0.3 Images/AF-200/CFD-AF-200-228-N.jpg'
    for l in os.listdir('./CFD 2.0.3 Images'):
        if '-' in l:
            for f in os.listdir('./CFD 2.0.3 Images/' + l):
                if f.endswith('N.jpg'):
                    save_cropped_image(
                        './CFD 2.0.3 Images/' + l + '/' + f,
                        './cropped/' + f
                    )
