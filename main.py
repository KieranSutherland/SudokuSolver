from cv2 import cv2
import tensorflow as tf
import traceback
import time
from predict import predict_grid_numbers
from image_processing import get_individual_number_imgs, get_grid_img, resize_img
from validator import is_grid_valid
from solver import solve
from image_solution_gen import generate_solution_grid_img, generate_final_solution_img
from utils import *
import copy

def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    model = tf.keras.models.load_model('digits_model')

    while True:
        try:
            if cv2.waitKey(1) == 27: # escape key
                break

            _, frame = camera.read()
            original_frame = frame.copy()
            grid_img, corners, destination = get_grid_img(frame)
            if grid_img is None:
                cv2.imshow('final_img', original_frame)
                continue
            
            grid_img_resized = resize_img(grid_img)
            grid_number_imgs = get_individual_number_imgs(grid_img_resized)
            # start = time.time()
            predicted_grid = predict_grid_numbers(model, grid_number_imgs)
            predicted_grid_original = copy.deepcopy(predicted_grid)
            # print("predict time: " + str(time.time() - start))

            if not is_grid_valid(predicted_grid):
                print("Grid is not valid, continuing to next loop and printing full grid for debug...")
                display_gameboard(predicted_grid)
                cv2.imshow('final_img', original_frame)
                continue

            calculate_accuracy_test_img(predicted_grid) # only for testing purposes

            solved_grid = solve(predicted_grid)
            if solved_grid is None:
                print("COULD NOT solve the puzzle!")
                cv2.imshow('final_img', original_frame)
                continue
            print("Solved the puzzle!")
            
            # grid excluding the numbers that were already there
            solved_grid_exc_predicted = exclude_predicted_nums(solved_grid, predicted_grid_original)
            # generate grid image of only the solved numbers
            solution_grid_img = generate_solution_grid_img(solved_grid_exc_predicted, grid_img)
            # merge the solved numbers grid image to the original frame image, masking them together
            final_solution_img = generate_final_solution_img(solution_grid_img, original_frame, corners, destination)
            cv2.imshow('final_img', final_solution_img)

        except Exception:
            cv2.imshow('final_img', original_frame)
            print(traceback.format_exc())

    clean_down(camera)

if __name__ == "__main__":
    main()