

def is_grid_valid(grid):
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
        if not is_group_valid(grid[i]): # row
            return False
        if not is_group_valid([grid[j][i] for j in range(9)]): # column
            return False
        if not is_group_valid(get_box_nums(grid, i)): # box
            return False

    return True


def get_box_nums(grid, box_num):
    """Return all numbers in the given box number (box_num). Must be between 0-8.
    Below is a grid labelling what grid numbers are assigned to what box:
    ┎─────┰─────┰─────┒
    ┃  0  ┃  1  ┃  2  ┃
    ┠─────╂─────╂─────┨
    ┃  3  ┃  4  ┃  5  ┃
    ┠─────╂─────╂─────┨
    ┃  6  ┃  7  ┃  8  ┃
    ┖─────┸─────┸─────┚
    """

    if box_num < 0 or box_num > 8:
        raise Exception('Invalid argument. Must be between 0-8')
    scr = (box_num // 3) * 3 # starting cell row
    scc = ((box_num + 3) % 3) * 3 # starting cell column
    return [
        grid[scr+0][scc+0], grid[scr+0][scc+1], grid[scr+0][scc+2], 
        grid[scr+1][scc+0], grid[scr+1][scc+1], grid[scr+1][scc+2], 
        grid[scr+2][scc+0], grid[scr+2][scc+1], grid[scr+2][scc+2]]
        

def is_group_valid(group_array):
    # print("validating: " + str(group_array))
    group_array_no_zeroes = list(filter(lambda num: num != 0, group_array))
    for num in group_array_no_zeroes:
        if num < 0 or num > 9:
            print("Found number below 0 or above 9!")
            return False
    return len(group_array_no_zeroes) == len(set(group_array_no_zeroes))