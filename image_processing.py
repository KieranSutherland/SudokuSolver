from cv2 import cv2
import numpy as np
from imutils import contours
import os
import time
from utils import display_image

def resize_img(img):
    img_copy = img.copy()
    height, width = img_copy.shape[:2]
    new_height = 252
    new_width = 252
    scaling_factor = new_height / float(height)
    if new_width/float(width) < scaling_factor:
        scaling_factor = new_width / float(width)
    return cv2.resize(img_copy, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

def get_individual_number_imgs(grid_img):
    edge_h = np.shape(grid_img)[0]
    edge_w = np.shape(grid_img)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid_img)[1] // 9

    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid_img[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])
    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])
    try:
        for i in range(9):
            for j in range(9):
                os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite("BoardCells/cell" + str(i) + str(j) + ".jpg", finalgrid[i][j])
    
    return finalgrid

def get_grid_img(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,4)
    # cv2.imshow("original", image)
    # display_image(image)

    contours = get_sorted_contours(image)
    contour, corners = get_grid_contour(contours)
    x, y, width, height = cv2.boundingRect(contour)
    average_grid_wh = int((int(width) + int(height)) / 2)
    
    # show outline of selected grid on full image
    # grid_outline = draw_grid_outline(frame.copy(), corners)
    # cv2.imshow("grid_outline", grid_outline)

    if contour is None:
        return None, None

    warped_grid, transform_matrix_inv = warp(image, corners, average_grid_wh)
    
    return warped_grid, transform_matrix_inv

def get_grid_contour(contours):
    for contour in contours:
        corners = get_contour_corners(contour)
        area = calculate_area(corners)
        # grid_outline_debug = draw_grid_outline(original_frame.copy(), corners)

        if area < 75000: 
            print("Too small")
            # cv2.imshow("original", image)
            # display_image(grid_outline_debug, "too small")
            return None, None

        (x, y, w, h) = cv2.boundingRect(contour.astype(np.int))

        if w / 1.35 > h or h / 1.35 > w:
            print("Not square enough")
            # cv2.imshow("original", image)
            # display_image(grid_outline_debug, "rectangle")
            continue

        # display_image(grid_outline_debug)
        return contour, corners
    print("Found 0 contours")
    return None, None

def calculate_area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def draw_grid_outline(image, corners):
    if corners is None or len(corners) < 4:
        return image
    grid_outline = image.copy()
    for i in range(3):
        cv2.line(grid_outline, tuple(corners[i]), tuple(corners[i+1]), (0, 255, 0), 3)
    cv2.line(grid_outline, tuple(corners[0]), tuple(corners[3]), (0, 255, 0), 3)
    return grid_outline

def warp(image, corners, warp_size):
    image_copy = image.copy()
    destination = np.array([[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(corners, destination)
    transform_matrix_inv = cv2.getPerspectiveTransform(destination, corners) # inverse to do opposite transformation later
    warped = cv2.warpPerspective(image_copy, transform_matrix, (warp_size, warp_size))
    return warped, transform_matrix_inv

def get_sorted_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    return contours[:5]

def get_contour_corners(contour):
    sums = []
    diffs = []
    for point in contour:
        for x, y in point:
            sums.append(x + y)
            diffs.append(x - y)
            
    top_left = contour[np.argmin(sums)].squeeze()
    bottom_right = contour[np.argmax(sums)].squeeze() 
    top_right = contour[np.argmax(diffs)].squeeze()
    bottom_left = contour[np.argmin(diffs)].squeeze() 
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)