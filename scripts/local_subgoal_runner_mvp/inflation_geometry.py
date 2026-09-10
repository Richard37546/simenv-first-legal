"""Pure, shared occupancy-inflation geometry for local navigation.

The mask is centred on occupied cell centres.  Unknown cells are deliberately
not seeds here: each caller preserves unknown blocking separately.
"""

from __future__ import annotations

import math

import numpy as np


INFLATION_GEOMETRY = "EUCLIDEAN_CELL_CENTER"


def occupied_euclidean_inflated_mask(
    grid: np.ndarray,
    robot_radius_m: float,
    resolution_m: float,
) -> np.ndarray:
    """Return raw-occupied cells expanded by a Euclidean cell-centre radius.

    ``ceil(radius / resolution)`` bounds the iteration only.  It is not an
    enlargement of the physical radius: a candidate cell is blocked exactly
    when its centre distance from an occupied seed centre is at most
    ``robot_radius_m``.
    """
    if grid.ndim != 2:
        raise ValueError("local_grid_must_be_2d")
    resolution = float(resolution_m)
    radius = float(robot_radius_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution_m_must_be_positive")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("robot_radius_m_must_be_nonnegative")

    occupied = np.asarray(grid) == 100
    inflated = occupied.copy()
    index_limit = int(math.ceil(radius / resolution))
    radius_in_cells_squared = (radius / resolution) * (radius / resolution)
    y_indices, x_indices = np.where(occupied)
    for y_index, x_index in zip(y_indices.tolist(), x_indices.tolist()):
        y0 = max(0, y_index - index_limit)
        y1 = min(grid.shape[0], y_index + index_limit + 1)
        x0 = max(0, x_index - index_limit)
        x1 = min(grid.shape[1], x_index + index_limit + 1)
        for candidate_y in range(y0, y1):
            for candidate_x in range(x0, x1):
                dy_cells = candidate_y - y_index
                dx_cells = candidate_x - x_index
                if dx_cells * dx_cells + dy_cells * dy_cells <= radius_in_cells_squared:
                    inflated[candidate_y, candidate_x] = True
    return inflated
