import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray.structures import visibilities as vis

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestVisibilitiesAPI:
    class TestManual:
        def test__visibilities__makes_visibilities_without_other_inputs(self):

            visibilities = aa.visibilities.manual_1d(
                visibilities=[[1.0, 2.0], [3.0, 4.0]]
            )

            assert type(visibilities) == vis.Visibilities
            assert visibilities.in_1d_flipped == np.array([[2.0, 1.0], [4.0, 3.0]])
            assert (visibilities.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (visibilities.real == np.array([1.0, 3.0])).all()
            assert (visibilities.imag == np.array([2.0, 4.0])).all()
            assert (visibilities.amplitudes == np.array([np.sqrt(5), 5.0])).all()
            assert visibilities.phases == pytest.approx(
                np.array([1.10714872, 0.92729522]), 1.0e-4
            )

            visibilities = aa.visibilities.manual_1d(
                visibilities=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            )

            assert type(visibilities) == vis.Visibilities
            assert (
                visibilities.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            ).all()
            assert (visibilities.real == np.array([1.0, 3.0, 5.0])).all()
            assert (visibilities.imag == np.array([2.0, 4.0, 6.0])).all()

    class TestFull:
        def test__visibilities__makes_visibilities_without_other_inputs(self):

            visibilities = aa.visibilities.full(fill_value=1.0, shape_1d=(2,))

            assert type(visibilities) == vis.Visibilities
            assert (visibilities.in_1d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

            visibilities = aa.visibilities.full(fill_value=2.0, shape_1d=(2,))

            assert type(visibilities) == vis.Visibilities
            assert (visibilities.in_1d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

    class TestOnesZeros:
        def test__visibilities__makes_visibilities_without_other_inputs(self):

            visibilities = aa.visibilities.ones(shape_1d=(2,))

            assert type(visibilities) == vis.Visibilities
            assert (visibilities.in_1d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

            visibilities = aa.visibilities.zeros(shape_1d=(2,))

            assert type(visibilities) == vis.Visibilities
            assert (visibilities.in_1d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

    class TestFromFits:
        def test__visibilities__makes_visibilities_without_other_inputs(self):

            visibilities = aa.visibilities.from_fits(
                file_path=test_data_dir + "3x2_ones.fits", hdu=0
            )

            assert type(visibilities) == vis.Visibilities
            assert (visibilities.in_1d == np.ones((3, 2))).all()

            visibilities = aa.visibilities.from_fits(
                file_path=test_data_dir + "3x2_twos.fits", hdu=0
            )

            assert type(visibilities) == vis.Visibilities
            assert (visibilities.in_1d == 2.0 * np.ones((3, 2))).all()


class TestVisibilities:
    class TestOutputToFits:
        def test__output_to_files(self):

            visibilities = aa.visibilities.from_fits(
                file_path=test_data_dir + "3x2_ones.fits", hdu=0
            )

            output_data_dir = "{}/../test_files/visibilities/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )
            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            visibilities.output_to_fits(file_path=output_data_dir + "visibilities.fits")

            visibilities_from_out = aa.visibilities.from_fits(
                file_path=output_data_dir + "visibilities.fits", hdu=0
            )

            assert (visibilities_from_out.in_1d == np.ones((3, 2))).all()
