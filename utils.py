from cv2 import cv2

def display_image(img, label="image"):
    cv2.imshow(str(label), img.copy())  # Display the image
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
        print("Board number resolution SUCCESSFUL")
    else:
        print("Board number resolution FAILED. See below for debug:")
        display_gameboard(grid)

def clean_down(capture):
    capture.release()
    cv2.destroyAllWindows()

def exclude_predicted_nums(solved_grid, predicted_grid):
    exclusively_solved_nums_grid = []
    for row in range(9):
        temp_row = []
        for column in range(9):
            predicted_cell_num = predicted_grid[row][column]
            if predicted_cell_num == 0:
                temp_row.append(solved_grid[row][column])
            else:
                temp_row.append(0)
        exclusively_solved_nums_grid.append(temp_row)
    return exclusively_solved_nums_grid

def convert_grid_to_key(grid):
    return ''.join(str(item) for innerlist in grid for item in innerlist)