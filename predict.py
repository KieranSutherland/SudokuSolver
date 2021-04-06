import tensorflow as tf
from cv2 import cv2
import numpy as np
from utils import display_image
import numpy as np

def predict_grid_numbers(model, grid_number_imgs):
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]
    processed_imgs_map = []
    for i in range(9):
        for j in range(9):
            image = grid_number_imgs[i][j]
            image = cv2.resize(image, (28, 28))

            cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if (x < 3 or y < 3 or h < 3 or w < 3):
                    continue
                cell_img = image[y:y + h, x:x + w]
                cell_img = scale_and_centre(cell_img, 120)

                processed_imgs_map.append([i, j, preprocess(model, cell_img)])
    
    if len(processed_imgs_map) == 0:
        return tmp_sudoku
    return assign_predictions_to_grid(processed_imgs_map, model, tmp_sudoku)

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

def preprocess(model, cell_img):
    image = cell_img.copy()
    # image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    return image.reshape(1, 28, 28, 1)

def assign_predictions_to_grid(processed_imgs_map, model, tmp_sudoku):
    processed_imgs = list([img_map[2] for img_map in processed_imgs_map])
    predicted_nums = model.predict(np.vstack(processed_imgs))
    predicted_nums = list([predicted_num.argmax() for predicted_num in predicted_nums])
    if len(processed_imgs_map) != len(predicted_nums):
        raise Exception("There are less predicted digits than there are preprocessed images. Something's gone wrong!")
    for i in range(len(processed_imgs_map)):
        tmp_sudoku[processed_imgs_map[i][0]][processed_imgs_map[i][1]] = predicted_nums[i]
    return tmp_sudoku