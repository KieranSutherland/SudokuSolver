from cv2 import cv2
import numpy as np
from PIL import ImageFont

def generate_solution_grid_img(solved_grid_exc_predicted, grid_img):
    # FONT = ImageFont.truetype("./Crashnumbering.ttf", 80)
    FONT = cv2.FONT_HERSHEY_DUPLEX
    CELL_WIDTH = grid_img.shape[1] // 9
    OFFSET = CELL_WIDTH // 15
    FONT_SCALE = 1.5
    # print("CELL_WIDTH: " + str(CELL_WIDTH))
    # print("OFFSET: " + str(OFFSET))
    transparent_img = np.zeros((grid_img.shape[1], grid_img.shape[0]), np.uint8)
    for row in range(9):
        for column in range(9):
            if solved_grid_exc_predicted[row][column] == 0:
                continue
            cell_num_str = str(solved_grid_exc_predicted[row][column])
            (text_height, text_width), _ = cv2.getTextSize(cell_num_str, FONT, fontScale = FONT_SCALE, thickness = 1)
            bottomLeft = CELL_WIDTH*column + (CELL_WIDTH - text_width) // 2 + OFFSET
            bottomRight = CELL_WIDTH*(row+1) - (CELL_WIDTH - text_height) // 2 + OFFSET
            transparent_img = cv2.putText(transparent_img, cell_num_str, (int(bottomLeft), int(bottomRight)), FONT, FONT_SCALE, 255, thickness = 1, lineType = cv2.LINE_AA)
    return transparent_img
    
def generate_final_solution_img(solution_grid_img, original_frame, transform_matrix_inv):
    solution_warped = _unwarp(solution_grid_img, original_frame, transform_matrix_inv)
    merged = _merge_pics(original_frame, solution_warped)
    return merged

def _unwarp(solution_grid_img, original_frame, transform_matrix_inv):
    width, height, _ = original_frame.shape
    warp = cv2.warpPerspective(solution_grid_img, transform_matrix_inv, (height, width))
    return warp

def _merge_pics(original_frame, solution_warped):
    ret, warp = cv2.threshold(solution_warped, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(warp)
    img1_bg = cv2.bitwise_and(original_frame, original_frame, mask=mask_inv)

    fg = cv2.cvtColor(solution_warped, cv2.COLOR_GRAY2BGR)
    fg[solution_warped > 0] = (255, 55, 0)
    dst = cv2.add(img1_bg, fg)
    return dst