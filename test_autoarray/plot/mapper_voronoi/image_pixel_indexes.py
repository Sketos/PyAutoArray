import autoarray as aa
import autoarray.plot as aplt
import numpy as np


grid_7x7 = aa.grid.uniform(shape_2d=(7, 7), pixel_scales=0.25)
grid_9 = aa.grid.manual_1d(
    grid=[
        [0.6, -0.3],
        [0.5, -0.8],
        [0.2, 0.1],
        [0.0, 0.5],
        [-0.3, -0.8],
        [-0.6, -0.5],
        [-0.4, -1.1],
        [-1.2, 0.8],
        [-1.5, 0.9],
    ],
    shape_2d=(3, 3),
    pixel_scales=1.0,
)
voronoi_grid = aa.grid_voronoi(
    grid_1d=grid_9,
    nearest_pixelization_1d_index_for_mask_1d_index=np.zeros(
        shape=grid_7x7.shape_1d, dtype="int"
    ),
)
voronoi_mapper = aa.mapper(grid=grid_7x7, pixelization_grid=voronoi_grid)

aplt.mapper_obj(mapper=voronoi_mapper, image_pixel_indexes=[3, 4])
aplt.mapper_obj(mapper=voronoi_mapper, image_pixel_indexes=[[3, 4]])
aplt.mapper_obj(mapper=voronoi_mapper, image_pixel_indexes=[[3, 4], [6]])
aplt.mapper_obj(mapper=voronoi_mapper, image_pixel_indexes=[[(0, 0), (0, 1)], [(2, 2)]])