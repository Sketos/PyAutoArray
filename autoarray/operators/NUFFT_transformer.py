import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
from scipy import interpolate
from pynufft import NUFFT_cpu

# NOTE : To import pynufft on COSMA.
#sys.path.append("/cosma/home/durham/dc-amvr1/.local/lib/python3.6/site-packages")

from autoarray import decorator_util

import finufftpy

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


class pynufft_Transformer(NUFFT_cpu):
    def __init__(self, uv_wavelengths, grid):
        super(pynufft_Transformer, self).__init__()
        self.uv_wavelengths = uv_wavelengths
        self.grid = grid

        # NOTE: The plan need only be initialized once
        self.initialize_plan()

        # ...
        self.shift = np.exp(
            -2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scales[0]/2.0 * units.arcsec.to(units.rad) * self.uv_wavelengths[:, 1]
                + self.grid.pixel_scales[0]/2.0 * units.arcsec.to(units.rad) * self.uv_wavelengths[:, 0]
            )
        )


    def initialize_plan(self, ratio=2, interpolation_kernel=(6, 6)):

        if not isinstance(ratio, int):
            ratio = int(ratio)

        # ... NOTE : The u,v coordinated should be given in the order ...
        visibilities_normalized = np.array([
            self.uv_wavelengths[:, 1] / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi,
            self.uv_wavelengths[:, 0] / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi
        ]).T

        # NOTE:
        self.plan(
            visibilities_normalized,
            self.grid.shape_2d,
            (ratio * self.grid.shape_2d[0], ratio * self.grid.shape_2d[1]),
            interpolation_kernel
        )


    def visibilities_from_image(self, image_in_2d):
        """
        ...
        """

        # NOTE: Flip the image the autolens produces.
        visibilities = self.forward(image_in_2d[::-1, :])

        # NOTE: This is nessecary given the
        visibilities *= self.shift

        return visibilities.real, visibilities.imag


    def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):

        real_transfomed_mapping_matrix = np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )
        imag_transfomed_mapping_matrix = np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):
            image = mapping_matrix[:, source_pixel_1d_index].reshape(
                self.grid.shape_2d[0],
                self.grid.shape_2d[1]
            )

            real_visibilities, imag_visibilities = self.visibilities_from_image(
                image_in_2d=image
            )

            real_transfomed_mapping_matrix[:, source_pixel_1d_index] = real_visibilities
            imag_transfomed_mapping_matrix[:, source_pixel_1d_index] = imag_visibilities

        return [real_transfomed_mapping_matrix, imag_transfomed_mapping_matrix]






class FINUFFT_Transformer:
    def __init__(self, uv_wavelengths, grid):

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
            10**-6.0,
            image_in_2d[:, ::-1] + 1j*np.zeros(shape=image_in_2d.shape)
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
            10**-6.0,
            mapping_matrix_reshaped
        )

        for i in range(visibilities.shape[1]):
            visibilities[:, i] *= self.shift

        return [visibilities.real, visibilities.imag]


if __name__ == "__main__":

    # ...
    n_pixels = 100

    # ...
    pixel_scale = 0.05 # NOTE: This is given in units of arcsec

    # ...
    grid = al.grid.uniform(
        shape_2d=(
            n_pixels,
            n_pixels
        ),
        pixel_scales=(
            pixel_scale,
            pixel_scale
        ),
        sub_size=1
    )
    # n_pixels_FFT = 2048
    # grid_FFT = al.grid.uniform(
    #     shape_2d=(
    #         n_pixels_FFT,
    #         n_pixels_FFT
    #     ),
    #     pixel_scales=(
    #         pixel_scale,
    #         pixel_scale
    #     ),
    #     sub_size=1
    # )


    # ...
    N = 1000
    u_min = - 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    u_max = + 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    v_min = - 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    v_max = + 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    uv_wavelengths = np.array([
        np.linspace(u_min, u_max, N),
        np.linspace(v_min, v_max, N),
    ]).T
    uv_wavelengths = np.array([
        np.random.uniform(u_min, u_max, N),
        np.random.uniform(v_min, v_max, N),
    ]).T


    # ...
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(
                -0.05,
                0.1),
            axis_ratio=0.8,
            phi=120.0,
            einstein_radius=1.0
        ),
    )
    source_galaxy = al.Galaxy(
        redshift=2.0,
        light=al.lp.EllipticalGaussian(
            centre=(
                0.0,
                0.0),
            axis_ratio=0.75,
            phi=40.0,
            intensity=1.0,
            sigma=0.1
        ),
    )
    tracer = al.Tracer.from_galaxies(
        galaxies=[
            lens_galaxy, source_galaxy
        ]
    )
    image = tracer.profile_image_from_grid(
        grid=grid
    )

    # ...
    tracer_inversion = al.Tracer.from_galaxies(
        galaxies=[
            lens_galaxy,
            al.Galaxy(
                redshift=2.0,
                pixelization=al.pix.VoronoiMagnification(
                    shape=(30, 30)
                ),
                regularization=al.reg.Constant(coefficient=1.0)
            )
        ]
    )
    mappers_of_planes = tracer_inversion.mappers_of_planes_from_grid(
        grid=grid,
        inversion_uses_border=False,
        preload_sparse_grids_of_planes=None,
    )
    mapper = mappers_of_planes[-1]



    """
    # # ...
    # start = time.time()
    # transformer = al.transformer(
    #     uv_wavelengths=uv_wavelengths,
    #     grid_radians=grid.in_radians,
    #     preload_transform=False
    # )
    # real_visibilities__from__dft_transformer = transformer.real_visibilities_from_image(image=image)
    # imag_visibilities__from__dft_transformer = transformer.imag_visibilities_from_image(image=image)
    # end = time.time()
    # print("dft:", end - start)

    image_in_2d = image.in_2d
    image_in_2d = np.random.randint(2, size=image.in_2d.shape)

    # ...
    pynufft_Transformer_obj = pynufft_Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid
    )
    start = time.time()
    real_visibilities__from__pynufft_transformer, imag_visibilities__from__pynufft_transformer = pynufft_Transformer_obj.visibilities_from_image(
        image_in_2d=image_in_2d
    )
    end = time.time()
    print("nufft:", end - start)

    start = time.time()
    finufft_Transformer_obj = FINUFFT_Transformer(uv_wavelengths=uv_wavelengths, grid=grid)
    visibilities__from__fipynufft_transformer = finufft_Transformer_obj.visibilities_from_image(image_in_2d=image_in_2d)
    end = time.time()
    print("finufft:", end - start)

    plt.figure()
    # plt.plot(
    #     real_visibilities__from__dft_transformer,
    #     imag_visibilities__from__dft_transformer,
    #     linestyle="None",
    #     marker="o",
    #     markersize=10,
    #     color="b"
    # )
    plt.plot(
        real_visibilities__from__pynufft_transformer,
        imag_visibilities__from__pynufft_transformer,
        linestyle="None",
        marker="o",
        markersize=6,
        color="r"
    )
    plt.plot(
        visibilities__from__fipynufft_transformer.real,
        visibilities__from__fipynufft_transformer.imag,
        linestyle="None",
        marker="*",
        markersize=4,
        color="g"
    )
    plt.show()
    """

    # start = time.time()
    # transformer = al.transformer(
    #     uv_wavelengths=uv_wavelengths,
    #     grid_radians=grid.in_radians,
    #     preload_transform=False
    # )
    # visibilities_from_dft = transformer.transformed_mapping_matrices_from_mapping_matrix(mapping_matrix=mapper.mapping_matrix)
    # end = time.time()
    # print("dft:", end - start)

    pynufft_Transformer_obj = pynufft_Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid
    )
    start = time.time()
    visibilities_from_pynufft_Transformer_obj = pynufft_Transformer_obj.transformed_mapping_matrices_from_mapping_matrix(mapping_matrix=mapper.mapping_matrix)
    end = time.time()
    print("nufft:", end - start)


    start = time.time()
    finufft_Transformer_obj = FINUFFT_Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid
    )
    visibilities_from_finufft_Transformer_obj = finufft_Transformer_obj.transformed_mapping_matrices_from_mapping_matrix(mapping_matrix=mapper.mapping_matrix)
    end = time.time()
    print("finufft:", end - start)

    #visibilities_from_dft = np.array(visibilities_from_dft)
    visibilities_from_pynufft_Transformer_obj = np.array(visibilities_from_pynufft_Transformer_obj)
    visibilities_from_finufft_Transformer_obj = np.array(visibilities_from_finufft_Transformer_obj)

    n = 0
    plt.figure()
    # plt.plot(
    #     visibilities_from_dft[0, :, n],
    #     visibilities_from_dft[1, :, n],
    #     linestyle="None",
    #     marker="o",
    #     markersize=10,
    #     color="b"
    # )
    plt.plot(
        visibilities_from_pynufft_Transformer_obj[0, :, n],
        visibilities_from_pynufft_Transformer_obj[1, :, n],
        linestyle="None",
        marker="o",
        markersize=6,
        color="r"
    )
    plt.plot(
        visibilities_from_finufft_Transformer_obj[0, :, n],
        visibilities_from_finufft_Transformer_obj[1, :, n],
        linestyle="None",
        marker="*",
        markersize=4,
        color="g"
    )
    plt.show()
