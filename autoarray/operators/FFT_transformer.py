import os
import numpy as np
from astropy import units, constants


# ...
if os.environ["HOME"] == "/Users/ccbh87":
    COSMA_HOME = os.environ["COSMA_HOME_local"]
    COSMA_DATA = os.environ["COSMA7_DATA_local"]
elif os.environ["HOME"] == "/cosma/home/durham/dc-amvr1":
    COSMA_HOME = os.environ["COSMA_HOME_host"]
    COSMA_DATA = os.environ["COSMA7_DATA_host"]

# ...
workspace_path = COSMA_DATA + "/workspace"

# ...
import autofit as af
af.conf.instance = af.conf.Config(
    config_path = workspace_path + "/config",
    output_path = workspace_path + "/output")
import autolens as al


class FFT_Transformer(object):
    def __init__(self, uv_wavelengths, grid, interpolator="RegularGridInterpolator"):
        super(FFT_Transformer, self).__init__()
        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.grid = grid

        # TODO: Check that the shape_2d and pixel scales are the same.
        # ...
        self.u_fft = np.fft.fftshift(
            np.fft.fftfreq(
                grid.shape_2d[0], grid.pixel_scales[0] * units.arcsec.to(units.rad)
            )
        )
        self.v_fft = np.fft.fftshift(
            np.fft.fftfreq(
                grid.shape_2d[1], grid.pixel_scales[1] * units.arcsec.to(units.rad)
            )
        )
        u_fft_meshgrid, v_fft_meshgrid = np.meshgrid(
            self.u_fft, self.v_fft
        )

        # ... This is nessesary due to the way the grid in autolens is set up.
        self.shift = np.exp(
            -2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scales[0]/2.0 * units.arcsec.to(units.rad) * u_fft_meshgrid
                + self.grid.pixel_scales[0]/2.0 * units.arcsec.to(units.rad) * v_fft_meshgrid
            )
        )

        # NOTE: The interpolator determines the method that is going to be used to iinterpolate the Fourier Transform on to the regular Fourier grid.
        self.interpolator = interpolator

        # ...
        if self.interpolator is None:
            raise ValueError
        elif self.interpolator == "RegularGridInterpolator":
            self.uv = np.array(
                list(
                    zip(
                        self.uv_wavelengths[:, 0],
                        self.uv_wavelengths[:, 1]
                    )
                )
            )
        else:
            raise ValueError



    def visibilities_from_image(self, image_in_2d):
        """
        Generate visibilities from an image (in this case the image was created using autolens).
        """

        # TODO: check that the input image is actually a 2D array, otherwise raise an error.

        # NOTE: The input image is flipped to account for the way autolens is generating images
        z_fft = np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(
                    image_in_2d[::-1, :]
                )
            )
        )

        # ...
        z_fft_shifted = z_fft * self.shift

        # ...
        if self.interpolator is None:
            raise ValueError
        elif self.interpolator == "RegularGridInterpolator":
            real_interp = interpolate.RegularGridInterpolator(
                (
                    self.u_fft,
                    self.v_fft
                ),
                z_fft_shifted.real.T,
                method="linear",
                bounds_error=False,
                fill_value=0.0
            )
            imag_interp = interpolate.RegularGridInterpolator(
                (
                    self.u_fft,
                    self.v_fft
                ),
                z_fft_shifted.imag.T,
                method="linear",
                bounds_error=False,
                fill_value=0.0
            )
        else:
            raise ValueError

        # ...
        real_visibilities = real_interp(self.uv)
        imag_visibilities = imag_interp(self.uv)

        return real_visibilities, imag_visibilities


    # def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):
    #     """
    #     ...
    #     """
    #
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
    #         real_visibilities, imag_visibilities = self.visibilities_from_image(
    #             image_in_2d=image
    #         )
    #
    #         real_transfomed_mapping_matrix[:, source_pixel_1d_index] = real_visibilities
    #         imag_transfomed_mapping_matrix[:, source_pixel_1d_index] = imag_visibilities
    #
    #     return [real_transfomed_mapping_matrix, imag_transfomed_mapping_matrix]

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


    # ...
    N = 100
    u_min = - 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    u_max = + 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    v_min = - 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    v_max = + 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    uv_wavelengths = np.array([
        np.linspace(u_min, u_max, N),
        np.linspace(v_min, v_max, N),
    ]).T

    # create a FFT_transformer object
    transformer = FFT_Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid
    )
