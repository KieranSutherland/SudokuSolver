

def isGridValid(grid):
    # if there's less than 17 numbers resolved, it's impossible to solve
    total_valid_nums = sum([sum([row.count(i) for i in range(1, 10)]) for row in grid])
    total_nums = sum(len(row) for row in grid)
    if total_valid_nums < 17:
        print("Total valid numbers was below 17. Not valid")
        return False
    if total_nums != 81:
        print("Total grid numbers was not 81. Not valid")
        return False

    # check if there are any duplicate numbers in a row, column or 3x3 grid
    for i in range(9):
        if not groupIsValid(grid[i]): # row
            return False
        column = [grid[j][i] for j in range(9)]
        if not groupIsValid(column): # column
            return False
        if not groupIsValid(getBoxNums(grid, i)): # box
            return False

    return True


def getBoxNums(grid, grid_num):
    """Return all numbers in the given grid number (grid_num).
    Below is a grid labelling what grid numbers are assigned to what box:
    ┎─────┰─────┰─────┒
    ┃  0  ┃  1  ┃  2  ┃
    ┠─────╂─────╂─────┨
    ┃  3  ┃  4  ┃  5  ┃
    ┠─────╂─────╂─────┨
    ┃  6  ┃  7  ┃  8  ┃
    ┖─────┸─────┸─────┚
    """
    box = []
    if grid_num == 0:
        box = [grid[0][0], grid[0][1], grid[0][2], grid[1][0], grid[1][1], grid[1][2], grid[2][0], grid[2][1], grid[2][2]]
    if grid_num == 1:
        box = [grid[0][3], grid[0][4], grid[0][5], grid[1][3], grid[1][4], grid[1][5], grid[2][3], grid[2][4], grid[2][5]]
    if grid_num == 2:
        box = [grid[0][6], grid[0][7], grid[0][8], grid[1][6], grid[1][7], grid[1][8], grid[2][6], grid[2][7], grid[2][8]]
    if grid_num == 3:
        box = [grid[3][0], grid[3][1], grid[3][2], grid[4][0], grid[4][1], grid[4][2], grid[5][0], grid[5][1], grid[5][2]]
    if grid_num == 4:
        box = [grid[3][3], grid[3][4], grid[3][5], grid[4][3], grid[4][4], grid[4][5], grid[5][3], grid[5][4], grid[5][5]]
    if grid_num == 5:
        box = [grid[3][6], grid[3][7], grid[3][8], grid[4][6], grid[4][7], grid[4][8], grid[5][6], grid[5][7], grid[5][8]]
    if grid_num == 6:
        box = [grid[6][0], grid[6][1], grid[6][2], grid[7][0], grid[7][1], grid[7][2], grid[8][0], grid[8][1], grid[8][2]]
    if grid_num == 7:
        box = [grid[6][3], grid[6][4], grid[6][5], grid[7][3], grid[7][4], grid[7][5], grid[8][3], grid[8][4], grid[8][5]]
    if grid_num == 8:
        box = [grid[6][6], grid[6][7], grid[6][8], grid[7][6], grid[7][7], grid[7][8], grid[8][6], grid[8][7], grid[8][8]]
    return box
            
def groupIsValid(group_array):
    print("Validating nums: " + str(group_array))
    group_array_no_zeroes = list(filter(lambda num: num != 0, group_array))
    for num in group_array_no_zeroes:
        if num < 0 or num > 9:
            print("Found number below 0 or above 9!")
            return False
    return len(group_array_no_zeroes) == len(set(group_array_no_zeroes))

