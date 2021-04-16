from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import time
from utils import convert_grid_to_key

class SudokuGrid:
    """
    A representation of the current state of a Sudoku grid.
    """

    def __init__(self, cells: List[List[int]], n=9,
                 cell_set={i for i in range(1, 10)}, grid_map=None) -> None:
        """
        Initializes the grid.
        Empty spaces are denoted by the 0 cell, and the grid cells
        are represented as letters or numerals.
        ===== Preconditions =====
        - n is an integer that is a perfect square
        - the given cell_set must contain n unique cells
        - for the grid to print properly, cells should be a 1 digit integer
        - there are n lists in cells, and each list has n integers as elements
        """
        # ===== Private Attributes =====
        # _n: The number of rows and columns in the grid.
        # _cells: A list of lists representing
        #           the current state of the grid.
        # _cell_set: The set of cells that each row, column, and subsquare
        #              must have exactly one of for this grid to be solved.
        # _map: A dictionary mapping each unfilled position to the possible
        #       cells that can still be filled in at that position.
        # _set_map: A dictionary that maps the unfilled cells for each
        #           row, column, and subsquare set to the possible positions
        #           that they could occupy within that section.

        _n: int
        _cells: List[List[int]]
        _cell_set: Set[int]
        _map: Dict[Tuple[int, int], Set[int]]
        _set_map: Dict[int, Dict[int, Set[Tuple[int, int]]]]

        assert n == len(cells), 'length of cells not equal to value of n'
        self._n, self._cells, self._cell_set, self._set_map = n, cells, cell_set, {}
        if grid_map is None:
            self._map = {}
            self._populate_map()
        else:
            self._map = grid_map

    def _populate_map(self) -> None:
        # updates _map with possible cells for each unfilled position
        for r in range(self._n):
            for c in range(self._n):
                if self._cells[r][c] == 0:
                    subset = self._row_set(r) | self._column_set(c) | self._subsquare_set(r, c)
                    allowed_cells = self._cell_set - subset
                    self._map[(r, c)] = allowed_cells

    def _populate_set_map(self) -> None:
        # updates _set_map with missing cells for each set
        # and the positions they could possibly occupy within the set
        for r in range(self._n):
            set_name = f'row{r}'
            self._set_map[set_name] = {}
            row_set = self._row_set(r)
            missing_cells = self._cell_set - row_set
            for sym in missing_cells:
                self._set_map[set_name][sym] = set()
                for key, value in self._map.items():
                    if key[0] == r and sym in value:
                        self._set_map[set_name][sym].add(key)
        if self._n > 9:
            for c in range(self._n):
                set_name = f'col{c}'
                self._set_map[set_name] = {}
                col_set = self._column_set(c)
                missing_cells = self._cell_set - col_set
                for sym in missing_cells:
                    self._set_map[set_name][sym] = set()
                    for key, value in self._map.items():
                        if key[1] == c and sym in value:
                            self._set_map[set_name][sym].add(key)
            n = round(self._n ** (1 / 2))
            for r in range(0, self._n, n):
                for c in range(0, self._n, n):
                    set_name = f'ss{r // n}{c // n}'
                    self._set_map[set_name] = {}
                    subsq_set = self._subsquare_set(r, c)
                    missing_cells = self._cell_set - subsq_set
                    for sym in missing_cells:
                        self._set_map[set_name][sym] = set()
                        for key, value in self._map.items():
                            if key[0] // n == r // n and key[1] // n == c // n and sym in value:
                                self._set_map[set_name][sym].add(key)

    def get_grid(self) -> List[List[int]]:
        """
        Returns a copy of the grid in a 2D array.
        """
        return [row for row in self._cells]

    def __str__(self) -> str:
        string_repr, n = [], round(self._n ** (1 / 2))
        div = '--' * n + ('+' + '-' + '--' * n) * (n - 2) + '+' + '--' * n
        for i in range(self._n):
            if i > 0 and i % n == 0:
                string_repr.append(div)
            row_lst = self._cells[i][:]
            for index in range(n, self._n, n + 1):
                row_lst.insert(index, '|')
            string_repr.append(' '.join(row_lst))
        return '\n'.join(string_repr)

    def is_solved(self) -> bool:
        return not any(0 in row for row in self._cells) \
            and self._check_row_and_col() and self._check_subsquares()

    def _check_row_and_col(self) -> bool:
        # (helper for is_solved)
        # checks that all rows and columns are filled in properly
        return all(self._row_set(i) == self._cell_set and
                   self._column_set(i) == self._cell_set
                   for i in range(self._n))

    def _check_subsquares(self) -> bool:
        # (helper for is_solved)
        # checks that all subsquares are filled in properly
        n = round(self._n ** (1 / 2))
        return all(self._subsquare_set(r, c) == self._cell_set
                   for r in range(0, self._n, n) for c in range(0, self._n, n))

    def extensions(self) -> List[SudokuGrid]:
        """
        Returns a list of SudokuGrid objects that have the position
        with the least number of possibilities filled in.
        This method checks for naked singles first, and if none are found,
        checks for hidden singles. Again, if none are found, it fills in the
        spot with the least number of naked/hidden possibilities.
        """
        if not self._map:
            return []
        extensions = []
        position, possible = None, self._cell_set | {0}
        for pos, values in self._map.items():
            if len(values) < len(possible):
                position, possible = pos, values
        cell, possible_positions = None, None
        if len(possible) > 1:
            self._populate_set_map()
            for d in self._set_map.values():
                for sym, positions in d.items():
                    if len(positions) < len(possible):
                        cell, possible_positions, = sym, positions
        if cell:
            for pos in possible_positions:
                new_cells = [row[:] for row in self._cells]
                new_cells[pos[0]][pos[1]] = cell
                new_map = self._map.copy()
                for key in self._get_positions(pos):
                    new_map[key] = self._map[key] - {cell}
                del new_map[pos]
                extensions.append(SudokuGrid(new_cells, self._n,
                                               self._cell_set, new_map))
        else:
            for value in possible:
                new_cells = [row[:] for row in self._cells]
                new_cells[position[0]][position[1]] = value
                new_map = self._map.copy()
                for key in self._get_positions(position):
                    new_map[key] = self._map[key] - {value}
                del new_map[position]
                extensions.append(SudokuGrid(new_cells, self._n,
                                               self._cell_set, new_map))
        return extensions

    def _get_positions(self, pos: tuple) -> List[Tuple[int, int]]:
        # returns the keys of sets in _map that may need to be altered
        n = round(self._n ** (1 / 2))
        return [key for key in self._map if key[0] == pos[0] or
                key[1] == pos[1] or (key[0] // n == pos[0] // n and
                                     key[1] // n == pos[1] // n)]

    def _row_set(self, r: int) -> Set[int]:
        # returns the set of cells of row r
        return set(self._cells[r])

    def _column_set(self, c: int) -> Set[int]:
        # returns the set of cells of column c
        return set(row[c] for row in self._cells)

    def _subsquare_set(self, r: int, c: int) -> Set[int]:
        # returns the set of cells of the subsquare that [r][c] belongs to
        n, cells = self._n, self._cells
        ss = round(n ** (1 / 2))
        ul_row = (r // ss) * ss
        ul_col = (c // ss) * ss
        return set(cells[ul_row + i][ul_col + j]
                   for i in range(ss) for j in range(ss))


def depth_first_solve(grid: SudokuGrid) -> Optional[SudokuGrid]:
    """
    An iterative depth first search to solve the grid.
    """
    if grid.is_solved():
        return grid
    grid_queue = grid.extensions()
    while grid_queue:
        current = grid_queue.pop()
        if current.is_solved():
            return current
        grid_queue.extend(current.extensions())
    return None


def solve(grid, previously_solved_grids):
    grid_key = convert_grid_to_key(grid)
    if grid_key in previously_solved_grids:
        return previously_solved_grids[grid_key]

    grid = SudokuGrid(grid)
    solution = depth_first_solve(grid)
    if solution:
        return solution.get_grid()
    else:
        return None
