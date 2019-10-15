import numpy as np
import pytest

import autoarray as aa

class TestRegions:

    def test__mask_2d_index_for_mask_1d_index__compare_to_array_util(self):

        mask = aa.mask([[True, True, True], [True, False, False], [True, True, False]])

        mask_2d_index_for_mask_1d_index = aa.mask_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert mask.regions._mask_2d_index_for_mask_1d_index == pytest.approx(
            mask_2d_index_for_mask_1d_index, 1e-4
        )

    def test__blurring_mask_for_psf_shape__compare_to_array_util(self):

        mask = aa.mask(
            [
                [True, True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )

        blurring_mask_via_util = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask=mask, kernel_shape=(3, 3)
        )

        blurring_mask = mask.regions.blurring_mask_from_kernel_shape(kernel_shape=(3, 3))

        assert (blurring_mask == blurring_mask_via_util).all()

    def test__edge_image_pixels__compare_to_array_util(self):
        mask = aa.mask(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        edge_pixels_util = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        mask.mapping = aa.Mapping(mask=mask)

        assert mask.regions._edge_1d_indexes == pytest.approx(edge_pixels_util, 1e-4)
        assert mask.regions._edge_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
        assert mask.regions._edge_2d_indexes[10] == pytest.approx(np.array([3, 3]), 1e-4)
        assert mask.regions._edge_1d_indexes.shape[0] == mask.regions._edge_2d_indexes.shape[0]

    def test__edge_mask(self):
        mask = aa.mask(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        assert (
                mask.regions.edge_mask
                == np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
        ).all()

    def test__border_image_pixels__compare_to_array_util(self):
        mask = aa.mask(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        border_pixels_util = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert mask.regions._border_1d_indexes == pytest.approx(border_pixels_util, 1e-4)
        assert mask.regions._border_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
        assert mask.regions._border_2d_indexes[10] == pytest.approx(np.array([3, 7]), 1e-4)
        assert mask.regions._border_1d_indexes.shape[0] == mask.regions._border_2d_indexes.shape[0]

    def test__border_mask(self):
        mask = aa.mask(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        assert (
                mask.regions.border_mask
                == np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
        ).all()

    def test__sub_border_1d_indexes__compare_to_array_util_and_numerics(self):

        mask = aa.mask(
            [
                [False, False, False, False, False, False, False, True],
                [False, True, True, True, True, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, False, True, False, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, True, True, True, True, False, True],
                [False, False, False, False, False, False, False, True],
            ], sub_size=2
        )

        sub_border_pixels_util = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert mask.regions._sub_border_1d_indexes == pytest.approx(
            sub_border_pixels_util, 1e-4
        )

        mask = aa.mask(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ], sub_size=2,
        )

        assert (
                mask.regions._sub_border_1d_indexes == np.array([0, 5, 9, 14, 23, 26, 31, 35])
    ).all()

    def test__mask_1d_index_for_sub_mask_1d_index__compare_to_util(self):
        mask = aa.mask(
            [[True, False, True], [False, False, False], [True, False, False]], sub_size=2
        )

        mask_1d_index_for_sub_mask_1d_index_util = aa.mask_util.mask_1d_index_for_sub_mask_1d_index_from_mask(
            mask=mask, sub_size=2
        )

        assert (
                mask.regions._mask_1d_index_for_sub_mask_1d_index
                == mask_1d_index_for_sub_mask_1d_index_util
        ).all()

    def test__sub_mask_2d_index_for_sub_mask_1d_index__compare_to_array_util(self):
        mask = aa.mask([[True, True, True], [True, False, False], [True, True, False]], sub_size=2)

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert mask.regions._sub_mask_2d_index_for_sub_mask_1d_index == pytest.approx(
            sub_mask_2d_index_for_sub_mask_1d_index, 1e-4
        )

    def test__sub_mask__is_mask_at_sub_grid_resolution(self):

        mask = aa.mask([[False, True], [False, False]], sub_size=2)

        assert (
                mask.regions.sub_mask_2d
                == np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        ).all()

        mask = aa.mask([[False, False, True], [False, True, False]], sub_size=2)

        assert (
                mask.regions.sub_mask_2d
                == np.array(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
            ]
        )
        ).all()