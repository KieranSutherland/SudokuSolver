
def solve(grid):
    empty_cell = find_empty(grid)
    if not empty_cell:  # if no more empty cells, puzzle complete
        return grid
    else:
        row, col = empty_cell

    for num in range(1, 10):
        if valid(grid, num, (row, col)):
            grid[row][col] = num

            if solve(grid):
                return grid

            grid[row][col] = 0

    return None


def valid(grid, num, pos):

    # Check row
    for i in range(len(grid[0])):
        if grid[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(grid)):
        if grid[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if grid[i][j] == num and (i, j) != pos:
                return False

    return True


def find_empty(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return i, j  # row, column

    return None
