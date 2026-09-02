from __future__ import annotations

import math

from torch import Tensor


def maximum_weight_assignment(scores: Tensor) -> list[int]:
    """Hungarian maximum-weight row-to-distinct-column assignment.

    Returns one column index per row. The implementation is O(R*C^2), has no
    SciPy dependency, and supports the benchmark's partial R <= C matrices.
    """
    if scores.ndim != 2:
        raise ValueError("assignment scores must have shape [rows, columns]")
    rows, columns = scores.shape
    if rows == 0 or rows > columns:
        raise ValueError("assignment requires 0 < rows <= columns")
    costs = (-scores.detach().cpu()).tolist()
    if any(not math.isfinite(value) for row in costs for value in row):
        raise ValueError("assignment scores must be finite")

    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - row_potential[row0] - column_potential[column]
                if current < minimum[column]:
                    minimum[column] = current
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("Hungarian assignment did not cover every row")
    return assignment
