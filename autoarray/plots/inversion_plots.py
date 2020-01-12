import autoarray as aa
import matplotlib

backend = aa.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters
from autoarray.operators.inversion import mappers


def subplot(
    inversion,
    mask=None,
    lines=None,
    positions=None,
    grid=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):



    plotter = plotter.plotter_with_new_output_filename(
        output_filename="image_and_mapper"
    )

    rows, columns, figsize_tool = plotter.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    ratio = float(
        (
            inversion.mapper.grid.scaled_maxima[1]
            - inversion.mapper.grid.scaled_minima[1]
        )
        / (
            inversion.mapper.grid.scaled_maxima[0]
            - inversion.mapper.grid.scaled_minima[0]
        )
    )

    if plotter.aspect is "square":
        aspect_inv = ratio
    elif plotter.aspect is "auto":
        aspect_inv = 1.0 / ratio
    elif plotter.aspect is "equal":
        aspect_inv = 1.0

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    reconstructed_image(
        inversion=inversion,
        mask=mask,
        lines=lines,
        positions=positions,
        grid=include.inversion_image_pixelization_grid_from_fit(),
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 2, aspect=float(aspect_inv))

    reconstruction(
        inversion=inversion,
        positions=None,
        lines=lines,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 3, aspect=float(aspect_inv))

    errors(
        inversion=inversion,
        positions=None,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 4, aspect=float(aspect_inv))

    residual_map(
        inversion=inversion,
        positions=None,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 5, aspect=float(aspect_inv))

    chi_squared_map(
        inversion=inversion,
        positions=None,
        include=include,
        plotter=plotter,
    )

    plt.subplot(rows, columns, 6, aspect=float(aspect_inv))

    regularization_weights(
        inversion=inversion,
        positions=None,
        include=include,
        plotter=plotter,
    )

    plotter.output.to_figure(structure=None)

    plt.close()


def individuals(
    inversion,
    lines=None,
    plot_inversion_reconstruction=False,
    plot_inversion_errors=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
    plot_inversion_interpolated_reconstruction=False,
    plot_inversion_interpolated_errors=False,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    if plot_inversion_reconstruction:

        reconstruction(
            inversion=inversion, include=include, plotter=plotter
        )

    if plot_inversion_errors:

        errors(inversion=inversion, include=include, plotter=plotter)

    if plot_inversion_residual_map:

        residual_map(
            inversion=inversion, include=include, plotter=plotter
        )

    if plot_inversion_normalized_residual_map:

        normalized_residual_map(
            inversion=inversion, include=include, plotter=plotter
        )

    if plot_inversion_chi_squared_map:

        chi_squared_map(
            inversion=inversion, include=include, plotter=plotter
        )

    if plot_inversion_regularization_weight_map:

        regularization_weights(
            inversion=inversion, include=include, plotter=plotter
        )

    if plot_inversion_interpolated_reconstruction:

        interpolated_reconstruction(
            inversion=inversion,
            lines=lines,
            include=include,
            plotter=plotter,
        )

    if plot_inversion_interpolated_errors:

        interpolated_errors(
            inversion=inversion,
            lines=lines,
            include=include,
            plotter=plotter,
        )


@plotters.set_labels
def reconstructed_image(
    inversion,
    mask=None,
    grid=None,
    lines=None,
    positions=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        array=inversion.mapped_reconstructed_image,
        mask=mask,
        lines=lines,
        points=positions,
        grid=grid,
        include_origin=include.origin,
    )


def plot_values(
    inversion,
    source_pixel_values,
    lines=None,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    if plotter.output.format is "fits":
        return

    plotter.setup_figure()

    if isinstance(inversion.mapper, mappers.MapperRectangular):

        reconstructed_pixelization = inversion.mapper.reconstructed_pixelization_from_solution_vector(
            solution_vector=source_pixel_values
        )

        plotter.array.plot(
            array=reconstructed_pixelization,
            lines=lines,
            points=positions,
            include_origin=include.origin,
        )

        plotter.plot_rectangular_mapper(
            mapper=inversion.mapper,
            image_pixels=image_pixels,
            source_pixels=source_pixels,
            include_centres=include.centres,
            include_grid=include.grid,
            include_border=include.border,
        )

        plotter.output.to_figure(array=reconstructed_pixelization)

    elif isinstance(inversion.mapper, mappers.MapperVoronoi):

        plotter.plot_voronoi_mapper(
            mapper=inversion.mapper,
            source_pixel_values=source_pixel_values,
            lines=lines,
            image_pixels=image_pixels,
            source_pixels=source_pixels,
            include_centres=include.inversion_centres,
            include_grid=include.inversion_grid,
            include_border=include.inversion_border,
        )

        plotter.output.to_figure(array=None)

    plotter.close_figure()


@plotters.set_labels
def reconstruction(
    inversion,
    origin=True,
    lines=None,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.reconstruction,
        origin=origin,
        lines=lines,
        positions=positions,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )


@plotters.set_labels
def errors(
    inversion,
    origin=True,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.errors,
        origin=origin,
        positions=positions,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )


@plotters.set_labels
def residual_map(
    inversion,
    origin=True,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.residual_map,
        origin=origin,
        positions=positions,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )


@plotters.set_labels
def normalized_residual_map(
    inversion,
    origin=True,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.normalized_residual_map,
        origin=origin,
        positions=positions,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )


@plotters.set_labels
def chi_squared_map(
    inversion,
    origin=True,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plot_values(
        inversion=inversion,
        source_pixel_values=inversion.chi_squared_map,
        origin=origin,
        positions=positions,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )


@plotters.set_labels
def regularization_weights(
    inversion,
    origin=True,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    regularization_weights = inversion.regularization.regularization_weights_from_mapper(
        mapper=inversion.mapper
    )

    plot_values(
        inversion=inversion,
        source_pixel_values=regularization_weights,
        origin=origin,
        positions=positions,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )


@plotters.set_labels
def interpolated_reconstruction(
    inversion,
    lines=None,
    positions=None,
    grid=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        array=inversion.interpolated_reconstruction_from_shape_2d(),
        lines=lines,
        points=positions,
        grid=grid,
        include_origin=include.origin,
    )


@plotters.set_labels
def interpolated_errors(
    inversion,
    lines=None,
    positions=None,
    grid=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        array=inversion.interpolated_errors_from_shape_2d(),
        lines=lines,
        points=positions,
        grid=grid,
        include_origin=include.origin,
    )
