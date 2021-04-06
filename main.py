from cv2 import cv2
import tensorflow as tf
import traceback
from predict import resolve_numbers
from image_processing import get_individual_number_imgs, get_grid_img
from solver import isGridValid
from utils import *

def main():

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    model = tf.keras.models.load_model('models/digits_model')

    try:
        while True:
            if cv2.waitKey(1) == 27: # escape key
                break

            try:
                grid_img = get_grid_img(capture)
            except:
                print("Failed to resolve grid image")
                continue

            if grid_img is None:
                continue
                
            cv2.imshow('grid', grid_img)
            grid_number_imgs = get_individual_number_imgs(grid_img)
            resolved_grid = resolve_numbers(model, grid_number_imgs)
            if not isGridValid(resolved_grid):
                print("Grid is not valid, continuing to next loop and printing full grid for debug...")
                display_gameboard(resolved_grid)
                continue

            # display_gameboard(resolved_grid)
            calculate_accuracy_test_img(resolved_grid)
    except Exception:
        print(traceback.format_exc())

    clean_down(capture)

if __name__ == "__main__":
    main()