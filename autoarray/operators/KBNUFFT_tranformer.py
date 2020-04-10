import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
from scipy import interpolate

# NOTE : To import pynufft on COSMA.
#sys.path.append("/cosma/home/durham/dc-amvr1/.local/lib/python3.6/site-packages")

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


from torchkbnufft import KbNufft
import torch


class kbnufft_Transformer:
    def __init__(self, uv_wavelengths, grid):

        self.dtype = torch.float

        self.uv_wavelengths = uv_wavelengths

        self.uv_wavelengths = np.array([
            self.uv_wavelengths[:, 1] / (1.0 / (2.0 * grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi,
            self.uv_wavelengths[:, 0] / (1.0 / (2.0 * grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi
        ]).T

        self.uv_wavelengths = np.stack(
            (
                np.transpose(uv_wavelengths[:, 0]).flatten(),
                np.transpose(uv_wavelengths[:, 1]).flatten()
            ),
            axis=0
        )
        self.uv_wavelengths = torch.tensor(
            self.uv_wavelengths
        ).to(self.dtype).unsqueeze(0)

        self.grid = grid

        # # NOTE: The plan need only be initialized once
        # self.initialize_plan()

        ratio = 2
        grid_size = (
            ratio * self.grid.shape_2d[0],
            ratio * self.grid.shape_2d[1]
        )

        self.nufft_ob = KbNufft(
            im_size=self.grid.shape_2d,
            grid_size=grid_size,
            norm='ortho'
        ).to(self.dtype)

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




    def initialize_plan(self, ratio=2):

        grid_size = (
            ratio * self.grid.shape_2d[0],
            ratio * self.grid.shape_2d[1]
        )

        self.nufft_ob = KbNufft(
            im_size=self.grid.shape_2d,
            grid_size=grid_size,
            norm='ortho'
        ).to(self.dtype)

    #     if not isinstance(ratio, int):
    #         ratio = int(ratio)
    #
    #     # ... NOTE : The u,v coordinated should be given in the order ...
    #     visibilities_normalized = np.array([
    #         self.uv_wavelengths[:, 1] / (1.0 / (2.0 * grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi,
    #         self.uv_wavelengths[:, 0] / (1.0 / (2.0 * grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi
    #     ]).T
    #
    #     # NOTE:
    #     self.plan(
    #         visibilities_normalized,
    #         self.grid.shape_2d,
    #         (ratio * self.grid.shape_2d[0], ratio * self.grid.shape_2d[1]),
    #         interpolation_kernel
    #     )


    def visibilities_from_image(self, image_in_2d):
        """
        ...
        """

        # NOTE: ...
        image_in_2d = image_in_2d[::-1, :]

        image_in_2d = np.stack((
            np.real(image_in_2d),
            np.imag(image_in_2d)
        ))
        image_in_2d = torch.tensor(
            image_in_2d
        ).to(self.dtype).unsqueeze(0).unsqueeze(0)

        print(image_in_2d.shape, self.uv_wavelengths.shape);exit()
        visibilities = self.nufft_ob(
            image_in_2d, self.uv_wavelengths
        )

        # # # NOTE: Flip the image the autolens produces.
        # # visibilities = self.forward(image_in_2d[::-1, :])
        #
        # # NOTE: This is nessecary given the
        # visibilities *= self.shift
        #
        # return visibilities.real, visibilities.imag


    # def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):
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

import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from torchkbnufft import KbNufft, AdjKbNufft
from torchkbnufft.mri.dcomp_calc import calculate_radial_dcomp_pytorch
from torchkbnufft.math import absolute

dtype = torch.float





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
    N = 500
    u_min = - 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    u_max = + 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    v_min = - 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    v_max = + 1.0 / (2.0 * pixel_scale * units.arcsec.to(units.rad))
    uv_wavelengths = np.array([
        np.linspace(u_min, u_max, N),
        np.linspace(v_min, v_max, N),
    ]).T
    #print(uv_wavelengths.shape);exit()

    shift = np.exp(
        -2.0
        * np.pi
        * 1j
        * (
            grid.pixel_scales[0]/2.0 * units.arcsec.to(units.rad) * uv_wavelengths[:, 1]
            + grid.pixel_scales[0]/2.0 * units.arcsec.to(units.rad) * uv_wavelengths[:, 0]
        )
    )


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

    # ========= #


    #_image = shepp_logan_phantom().astype(np.complex)

    _image = image.in_2d[::-1, :]
    _image = _image.astype(np.complex)
    im_size = _image.shape
    # plt.imshow(_image)
    # plt.gray()
    # plt.show()
    # exit()



    # convert the phantom to a tensor and unsqueeze coil and batch dimension
    _image = np.stack((np.real(_image), np.imag(_image)))
    _image = torch.tensor(_image).to(dtype).unsqueeze(0).unsqueeze(0)

    spokelength = _image.shape[-1] * 2
    grid_size = (spokelength, spokelength)
    nspokes = 100

    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    #print(np.min(kx), np.max(kx));exit()
    ky = np.transpose(ky)
    kx = np.transpose(kx)
    #print(ky.flatten().shape);exit()
    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    ktraj = torch.tensor(ktraj).to(dtype).unsqueeze(0)
    #print(ktraj.shape)#;exit()

    # uv_wavelengths_temp = np.array([
    #     uv_wavelengths[:, 1] / (1.0 / (2.0 * grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi,
    #     uv_wavelengths[:, 0] / (1.0 / (2.0 * grid.pixel_scales[0] * units.arcsec.to(units.rad))) * np.pi
    # ]).T
    #
    # uv_wavelengths_temp = np.stack(
    #     (
    #         np.transpose(uv_wavelengths_temp[:, 0]).flatten(),
    #         np.transpose(uv_wavelengths_temp[:, 1]).flatten()
    #     ),
    #     axis=0
    # )
    #
    # ktraj = torch.tensor(uv_wavelengths_temp).to(dtype).unsqueeze(0)


    nufft_ob = KbNufft(im_size=im_size, grid_size=grid_size, norm='ortho').to(dtype)
    adjnufft_ob = AdjKbNufft(im_size=im_size, grid_size=grid_size, norm='ortho').to(dtype)

    # calculate k-space data
    #print(image.shape, ktraj.shape);exit()
    kdata = nufft_ob(_image, ktraj)
    #print(kdata[0, 0, 0, :])
    #exit()

    # real_visibilities__from__kbnufft_transformer = kdata[0, 0, 0, :]
    # real_visibilities__from__kbnufft_transformer *= shift.real
    # imag_visibilities__from__kbnufft_transformer = kdata[0, 0, 1, :]
    # imag_visibilities__from__kbnufft_transformer *= shift.imag

    # image_blurry = adjnufft_ob(kdata, ktraj)
    #
    # # show the images
    # image_blurry_numpy = np.squeeze(image_blurry.numpy())
    # image_blurry_numpy = image_blurry_numpy[0] + 1j*image_blurry_numpy[1]
    #
    # plt.figure()
    # plt.imshow(np.absolute(image_blurry_numpy))
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(image.in_2d[::-1, :])
    # plt.colorbar()
    # plt.show()
    # exit()

    # ========= #

    # ...
    transformer = al.transformer(
        uv_wavelengths=uv_wavelengths,
        grid_radians=grid.in_radians,
        preload_transform=False
    )
    real_visibilities__from__dft_transformer = transformer.real_visibilities_from_image(image=image)
    imag_visibilities__from__dft_transformer = transformer.imag_visibilities_from_image(image=image)

    image_1d = transformer.image_from_visibilities(
        real_visibilities=real_visibilities__from__dft_transformer,
        imag_visibilities=imag_visibilities__from__dft_transformer
    )
    image_2d = image_1d.reshape(grid.shape_2d)
    plt.figure()
    plt.imshow(image_2d)
    plt.show()
    exit()

    # # ...
    # kbnufft_Transformer_obj = kbnufft_Transformer(
    #     uv_wavelengths=uv_wavelengths, grid=grid
    # )
    #
    # #real_visibilities__from__pynufft_transformer, imag_visibilities__from__pynufft_transformer =
    # kbnufft_Transformer_obj.visibilities_from_image(
    #     image_in_2d=image.in_2d
    # )
    #
    # exit()


    plt.figure()
    plt.plot(
        real_visibilities__from__dft_transformer,
        imag_visibilities__from__dft_transformer,
        linestyle="None",
        marker="o",
        markersize=10,
        color="b"
    )
    plt.plot(
        real_visibilities__from__kbnufft_transformer,
        imag_visibilities__from__kbnufft_transformer,
        linestyle="None",
        marker="o",
        markersize=5,
        color="r"
    )
    plt.show()
