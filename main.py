from cv2 import cv2
import tensorflow as tf
import traceback
import time
from predict import predict_grid_numbers
from image_processing import get_individual_number_imgs, get_grid_img
from validator import is_grid_valid
from solver import solve
from utils import *

def main():

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    model = tf.keras.models.load_model('digits_model')

    try:
        while True:
            if cv2.waitKey(1) == 27: # escape key
                break

            try:
                grid_img = get_grid_img(camera)
            except:
                print(traceback.format_exc())
                print("Failed to resolve grid image")
                continue

            if grid_img is None:
                continue
            
            # cv2.imshow('grid', grid_img)
            grid_number_imgs = get_individual_number_imgs(grid_img)
            # start = time.time()
            predicted_grid = predict_grid_numbers(model, grid_number_imgs)
            # print("predict time: " + str(time.time() - start))
            if not is_grid_valid(predicted_grid):
                print("Grid is not valid, continuing to next loop and printing full grid for debug...")
                display_gameboard(predicted_grid)
                continue

            calculate_accuracy_test_img(predicted_grid) # only for testing purposes

            solved_grid = solve(predicted_grid)
            if solved_grid is None:
                print("COULD NOT solve the puzzle!")
                continue
            # display_gameboard(predicted_grid)
            print("Solved the puzzle!")
    except Exception:
        print(traceback.format_exc())

    clean_down(camera)

if __name__ == "__main__":
    main()