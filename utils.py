from cv2 import cv2

def display_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows

def display_gameboard(sudoku):
    for i in range(len(sudoku)):
        if i % 3 == 0:
            if i == 0:
                print(" ┎─────────┰─────────┰─────────┒")
            else:
                print(" ┠─────────╂─────────╂─────────┨")

        for j in range(len(sudoku[0])):
            if j % 3 == 0:
                print(" ┃ ", end=" ")

            if j == 8:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", " ┃")
            else:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", end=" ")

    print(" ┖─────────┸─────────┸─────────┚")

def calculate_accuracy_test_img(grid):
    actual_grid = [
        [2, 5, 0, 0, 0, 3, 0, 9, 1], 
        [3, 0, 9, 0, 0, 0, 7, 2, 0], 
        [0, 0, 1, 0, 0, 6, 3, 0, 0], 
        [0, 0, 0, 0, 6, 8, 0, 0, 3], 
        [0, 1, 0, 0, 4, 0, 0, 0, 0], 
        [6, 0, 3, 0, 0, 0, 0, 5, 0], 
        [1, 3, 2, 0, 0, 0, 0, 7, 0], 
        [0, 0, 0, 0, 0, 4, 0, 6, 0], 
        [7, 6, 4, 0, 1, 0, 0, 0, 0]]
    if actual_grid == grid:
        return "Board number resolution SUCCESSFUL"
    else:
        return "Board number resolution FAILED"

def clean_down(capture):
    capture.release()
    cv2.destroyAllWindows()