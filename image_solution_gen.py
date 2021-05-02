from cv2 import cv2
import numpy as np

def generate_solution_grid_img(solved_grid_exc_predicted, grid_img):
    FONT = cv2.FONT_HERSHEY_DUPLEX
    CELL_WIDTH = grid_img.shape[1] // 9
    OFFSET = CELL_WIDTH // 15
    # FONT_SCALE = get_optimal_font_scale(CELL_WIDTH)
    FONT_SCALE = CELL_WIDTH / 10 / 4
    transparent_img = np.zeros((grid_img.shape[1], grid_img.shape[0]), np.uint8)
    (text_height, text_width), _ = cv2.getTextSize("0", FONT, fontScale = FONT_SCALE, thickness = 1)
    for row in range(9):
        for column in range(9):
            if solved_grid_exc_predicted[row][column] == 0:
                continue
            cell_num_str = str(solved_grid_exc_predicted[row][column])
            bottom_left_x = CELL_WIDTH * column + (CELL_WIDTH - text_width) // 2 + OFFSET
            bottom_left_y = CELL_WIDTH * (row + 1) - (CELL_WIDTH - text_height) // 2 + OFFSET
            transparent_img = cv2.putText(transparent_img, cell_num_str, (int(bottom_left_x), int(bottom_left_y)), FONT, FONT_SCALE, 255, thickness = 1, lineType = cv2.LINE_AA)
    return transparent_img
    

def get_optimal_font_scale(width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize("0", fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 10 / 2
    return 1


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
    background = cv2.bitwise_and(original_frame, original_frame, mask=mask_inv)

    foreground = cv2.cvtColor(solution_warped, cv2.COLOR_GRAY2BGR)
    foreground[solution_warped > 0] = (255, 55, 0)
    dst = cv2.add(background, foreground)
    return dst