from os import path
import os

import pytest

import autoarray as aa
import autoarray.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_interferometer_plotter_setup():
    plot_path = "{}/../../test_files/plotting/interferometer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return plot_path


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__individual_attributes_are_output(interferometer_7, plot_path, plot_patch):

    aplt.interferometer.visibilities(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "visibilities.png" in plot_patch.paths

    aplt.interferometer.noise_map(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "noise_map.png" in plot_patch.paths

    aplt.interferometer.u_wavelengths(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "u_wavelengths.png" in plot_patch.paths

    aplt.interferometer.v_wavelengths(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "v_wavelengths.png" in plot_patch.paths

    aplt.interferometer.uv_wavelengths(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "uv_wavelengths.png" in plot_patch.paths

    aplt.interferometer.amplitudes_vs_uv_distances(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "amplitudes_vs_uv_distances.png" in plot_patch.paths

    aplt.interferometer.phases_vs_uv_distances(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "phases_vs_uv_distances.png" in plot_patch.paths

    aplt.interferometer.primary_beam(
        interferometer=interferometer_7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "primary_beam.png" in plot_patch.paths


def test__subplot_is_output(interferometer_7, plot_path, plot_patch):

    aplt.interferometer.subplot_interferometer(
        interferometer=interferometer_7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "subplot_interferometer.png" in plot_patch.paths


def test__individuals__output_dependent_on_input(
    interferometer_7, plot_path, plot_patch
):
    aplt.interferometer.individual(
        interferometer=interferometer_7,
        plot_visibilities=True,
        plot_u_wavelengths=False,
        plot_v_wavelengths=True,
        plot_primary_beam=True,
        plot_amplitudes_vs_uv_distances=True,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "visibilities.png" in plot_patch.paths

    assert not plot_path + "u_wavelengths.png" in plot_patch.paths

    assert plot_path + "v_wavelengths.png" in plot_patch.paths

    assert plot_path + "amplitudes_vs_uv_distances.png" in plot_patch.paths

    assert plot_path + "phases_vs_uv_distances.png" not in plot_patch.paths

    assert plot_path + "primary_beam.png" in plot_patch.paths
