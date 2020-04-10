import numpy as np
from astropy import units

import finufftpy

# NOTE: This is added because I have two version of OpenMP on my laptop
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ...
autolens_version = "0.40.0"

# ...
if os.environ["HOME"] == "/Users/ccbh87":
    COSMA_HOME = os.environ["COSMA_HOME_local"]
    COSMA_DATA = os.environ["COSMA7_DATA_local"]
elif os.environ["HOME"] == "/cosma/home/durham/dc-amvr1":
    COSMA_HOME = os.environ["COSMA_HOME_host"]
    COSMA_DATA = os.environ["COSMA7_DATA_host"]

# ...
workspace_HOME_path = COSMA_HOME + "/workspace"
workspace_DATA_path = COSMA_DATA + "/workspace"

# ...
import autofit as af
af.conf.instance = af.conf.Config(
    config_path=workspace_DATA_path + "/config" + "_" + autolens_version,
    output_path=workspace_DATA_path + "/output")
import autolens as al


class FINUFFT_Transformer:
    def __init__(self, uv_wavelengths, grid, eps=10**-6.0):

        self.uv_wavelengths = uv_wavelengths

        self.grid = grid

        self.shift = np.exp(
            2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scale/2.0 * units.arcsec.to(units.rad) * self.uv_wavelengths[:, 1]
                + self.grid.pixel_scale/2.0 * units.arcsec.to(units.rad) * self.uv_wavelengths[:, 0]
            )
        )

        # NOTE: normalize the uv_wavelengths according to the max wavenumber given the pixel scale of the image
        self.uv_wavelengths_normalized = self.uv_wavelengths / (1.0 / (2.0 * self.grid.pixel_scale * units.arcsec.to(units.rad)))

        self.eps = eps


    def visibilities_from_image(self, image_in_2d):

        visibilities = np.zeros(
            shape=self.uv_wavelengths.shape[0],
            dtype=np.complex
        )
        ret = finufftpy.nufft2d2(
            self.uv_wavelengths_normalized[:, 1] * np.pi,
            self.uv_wavelengths_normalized[:, 0] * np.pi,
            visibilities,
            1,
            self.eps,
            image_in_2d[:, ::-1]
        )
        visibilities *= self.shift

        return visibilities

    # def transformed_mapping_matrices_from_mapping_matrix_slow(self, mapping_matrix):
    #
    #     # NOTE: This gives correct results but it's very slow ...
    #     real_transfomed_mapping_matrix = np.zeros(
    #         (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
    #     )
    #     imag_transfomed_mapping_matrix = np.zeros(
    #         (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
    #     )
    #
    #     for source_pixel_1d_index in range(mapping_matrix.shape[1]):
    #         image = mapping_matrix[:, source_pixel_1d_index].reshape(
    #             self.grid.shape_2d[0],
    #             self.grid.shape_2d[1]
    #         )
    #
    #         visibilities = self.visibilities_from_image(
    #             image_in_2d=image
    #         )
    #
    #         real_transfomed_mapping_matrix[:, source_pixel_1d_index] = visibilities.real
    #         imag_transfomed_mapping_matrix[:, source_pixel_1d_index] = visibilities.imag
    #
    #     return [real_transfomed_mapping_matrix, imag_transfomed_mapping_matrix]


    @staticmethod
    def reshape_mapping_matrix(mapping_matrix, shape_2d):

        mapping_matrix_reshaped = np.zeros(
            shape=(
                shape_2d + (mapping_matrix.shape[1],)
            )
        )

        for i in range(mapping_matrix.shape[1]):
            image = np.reshape(
                a=mapping_matrix[:, i], newshape=shape_2d
            )
            mapping_matrix_reshaped[:, :, i] = image[:, ::-1]

        return mapping_matrix_reshaped


    def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):

        mapping_matrix_reshaped = self.reshape_mapping_matrix(
            mapping_matrix=mapping_matrix,
            shape_2d=self.grid.shape_2d
        )

        visibilities = np.zeros(
            shape=(self.uv_wavelengths.shape[0], mapping_matrix.shape[1]),
            order='F',
            dtype=np.complex
        )
        finufftpy.nufft2d2many(
            self.uv_wavelengths_normalized[:, 1] * np.pi,
            self.uv_wavelengths_normalized[:, 0] * np.pi,
            visibilities,
            1,
            self.eps,
            mapping_matrix_reshaped
        )

        for i in range(visibilities.shape[1]):
            visibilities[:, i] *= self.shift

        return [visibilities.real, visibilities.imag]
