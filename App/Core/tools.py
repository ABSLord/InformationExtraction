import os
from .config import *
import numpy as np
import pandas as pd
from PIL import Image
import cv2

########################################################
#  Всякие вспомогательные функции
########################################################

def string_to_dataframe(string):
    data = list(map(lambda val: val.split('\t'), string.split('\n')))
    return pd.DataFrame(data[1:], columns=data[0])

def image_to_dataframe(path):
    img = Image.open(path)
    frame = string_to_dataframe(pytesseract.image_to_data(img1, lang=LANG))
    return frame

def extract_from_img(path, x, y, w, h):
    #  'data3/01.png', 1264, 870, 600, 100
    img1 = Image.open(path)
    img2 = extract_data_from_img(path, x, y, w, h)
    frame1 = string_to_dataframe(pytesseract.image_to_data(img1, lang=LANG))
    frame2 = string_to_dataframe(pytesseract.image_to_data(img2, lang=LANG))
    return frame1, frame2


def to_vector(y, l):
    arr = np.zeros([y.shape[0], l])
    for i in range(0, y.shape[0]):
        arr[i, int(y[i])] = 1
    return arr


def predicate(r, g, b, c, e):
    return (r >= c[0] - e) & (r <= c[0] + e) & \
           (g >= c[1] - e) & (g <= c[1] + e) & \
           (b >= c[2] - e) & (b <= c[2] + e)


def to_nparray(img):
    arr = np.array(Image.open(img).convert('RGB'))
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    result = np.zeros((arr.shape[0], arr.shape[1]))
    result[predicate(r, g, b, (255, 255, 255), 1)] = 0
    result[~predicate(r, g, b, (255, 255, 255), 1)] = 1
    result = result.reshape((result.shape[0] * result.shape[1],))
    return result


def to_bw(img):
    im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(img, im_bw)


def prepare_dataset(path, size, count_of_examples):
    x = np.zeros((count_of_examples, size))
    y = np.zeros((count_of_examples, ))
    i = -1
    for file in os.listdir(path):
        img = os.path.join(path, file)
        i += 1
        x[i] = to_nparray(img)
        y[i] = 0
    return x, y


def extract_data_from_img(img, x, y, w, h):
    image_file = Image.open(img)
    image_file = image_file.convert('L')
    image_file.save('result.png')
    img = cv2.imread('result.png')
    crop_img = img[y:y + h, x:x + w]
    return crop_img