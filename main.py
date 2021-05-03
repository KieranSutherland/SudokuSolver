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
    predicted_grid_original = None
    previously_solved_grids = dict()

    while True:
        try:
            if cv2.waitKey(1) == 27: # escape key
                break
            # time.sleep(1)
            _, frame = camera.read()
            # frame = cv2.imread('example_frames/example_hard.jpg') # here for testing, delete line for production
            if frame is None:
                print("No camera detected")
                continue
            original_frame = frame.copy()
            grid_img, transform_matrix_inv = get_grid_img(frame)
            if grid_img is None:
                cv2.imshow('final_img', original_frame)
                yield convert_frame_to_jpg(original_frame) # these yeilds should be commented out if wanting to run without server
                continue
            
            grid_img_resized = resize_img(grid_img)
            grid_number_imgs = get_individual_number_imgs(grid_img_resized)
            # start = time.time()
            predicted_grid = predict_grid_numbers(model, grid_number_imgs)
            # print("1 predict time: " + str(time.time() - start))
            # start = time.time()

            if not is_grid_valid(predicted_grid):
                print("Grid is not valid, continuing to next loop and printing full grid for debug:")
                display_gameboard(predicted_grid)
                cv2.imshow('final_img', original_frame)
                yield convert_frame_to_jpg(original_frame)
                continue
            
            # calculate_accuracy_test_img(predicted_grid) # only for testing purposes

            if predicted_grid_original == None or predicted_grid != predicted_grid_original: # if grid is the same, no need to solve again
                predicted_grid_original = copy.deepcopy(predicted_grid)
                grid_key = convert_grid_to_key(predicted_grid)
                solved_grid = solve(predicted_grid, grid_key, previously_solved_grids)
                # print("2 predict time: " + str(time.time() - start))
                # start = time.time()
                previously_solved_grids[grid_key] = solved_grid
                if solved_grid is None:
                    print("COULD NOT solve the puzzle! Printing full grid for debug:")
                    display_gameboard(predicted_grid_original)
                    cv2.imshow('final_img', original_frame)
                    yield convert_frame_to_jpg(original_frame)
                    continue
                print("Solved the puzzle!")
                # grid excluding the numbers that were already there
                solved_grid_exc_predicted = exclude_predicted_nums(solved_grid, predicted_grid_original)
            else:
                print("Same grid as previous loop, skipping some steps for optimisation")
                if solved_grid_exc_predicted == None:
                    solved_grid_exc_predicted = exclude_predicted_nums(solved_grid, predicted_grid_original)
            
            # generate grid image of only the solved numbers
            solution_grid_img = generate_solution_grid_img(solved_grid_exc_predicted, grid_img)
            # merge the solved numbers grid image to the original frame image, masking them together
            final_solution_img = generate_final_solution_img(solution_grid_img, original_frame, transform_matrix_inv)
            cv2.imshow('final_img', final_solution_img)
            yield convert_frame_to_jpg(final_solution_img)

        except Exception:
            cv2.imshow('final_img', original_frame)
            yield convert_frame_to_jpg(original_frame)
            print(traceback.format_exc())

    clean_down(camera)

if __name__ == "__main__":
    main()