import autoarray as aa
import autoarray.plot as aplt
import numpy as np

grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)

aplt.grid(grid=grid, symmetric_around_centre=True)

grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=1.0, origin=(10.0, 10.0))

aplt.grid(grid=grid, symmetric_around_centre=False)
aplt.grid(grid=grid, symmetric_around_centre=True)
