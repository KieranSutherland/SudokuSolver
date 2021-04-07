from cv2 import cv2
import numpy as np
from imutils import contours
import os
import time
from utils import display_image

def resize_img(img):
    height, width = img.shape[:2]
    new_height = 252
    new_width = 252
    scaling_factor = new_height / float(height)
    if new_width/float(width) < scaling_factor:
        scaling_factor = new_width / float(width)
    return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

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

def get_grid_img(camera):
    _, frame = camera.read()
    original_frame = frame.copy()
    # frame = cv2.imread('example_easy.jpg') # here for testing, delete line for production
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,4)
    # cv2.imshow("original", image)
    # display_image(image)

    coords = get_coords(image)

    # show outline of selected grid on full image
    grid_outline = draw_grid_outline(original_frame, coords)
    cv2.imshow("grid_outline", grid_outline)

    transformed = perspective_transform(image, coords)
    return resize_img(transformed)

def draw_grid_outline(image, coords):
    for i in range(3):
        cv2.line(image, tuple(coords[i]), tuple(coords[i+1]), (0, 255, 0), 3)
    cv2.line(image, tuple(coords[0]), tuple(coords[3]), (0, 255, 0), 3)
    return image

def perspective_transform(image, coords):
    ratio = 1.2
    tl, tr, br, bl = coords
    widthA = np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[1])**2)
    widthB = np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[1])**2)
    # heightA = np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[1])**2)
    # heightB = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[1])**2)
    width = max(widthA, widthB) * ratio
    height = width
    
    destination = np.array([
        [0, 0],
        [height, 0],
        [height, width],
        [0, width]], dtype = np.float32)
    M = cv2.getPerspectiveTransform(coords, destination)
    warped = cv2.warpPerspective(image, M, (int(height), int(width)))
    return warped

def get_coords(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    polygon = all_contours[0]
    sums = []
    diffs = []
    
    for point in polygon:
        for x, y in point:
            sums.append(x + y)
            diffs.append(x - y)
            
    top_left = polygon[np.argmin(sums)].squeeze()
    bottom_right = polygon[np.argmax(sums)].squeeze() 
    top_right = polygon[np.argmax(diffs)].squeeze()
    bottom_left = polygon[np.argmin(diffs)].squeeze() 
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)