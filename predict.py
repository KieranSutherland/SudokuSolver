import tensorflow as tf
from cv2 import cv2
import numpy as np
from utils import display_image

def predict_grid_numbers(model, grid_number_imgs):
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):

            image = grid_number_imgs[i][j]
            image = cv2.resize(image, (28, 28))

            # thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
            # image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

            # Find contours
            cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                if (x < 3 or y < 3 or h < 3 or w < 3):
                    # Note the number is always placed in the center
                    # Since image is 28x28
                    # the number will be in the center thus x >3 and y>3
                    # Additionally any of the external lines of the sudoku will not be thicker than 3
                    continue
                cell_img = image[y:y + h, x:x + w]
                cell_img = scale_and_centre(cell_img, 120)

                tmp_sudoku[i][j] = predict(model, cell_img)

    return tmp_sudoku

def scale_and_centre(img, size, margin=20, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

def predict(model, cell_img):
    image = cell_img.copy()
    # image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    display_img = image
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    # print(str(pred))
    # print("predction: " + str(pred.argmax()))
    # display_image(display_img)
    return pred.argmax()