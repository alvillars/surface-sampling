import h5py
import numpy as np 
from skimage.morphology import cube, binary_erosion
from morphometrics.utils.surface_utils import binary_mask_to_surface
from morphometrics.utils.image_utils import make_boundary_mask
import napari
import matplotlib.pyplot as plt 
import tifffile
from _utils_surface_sampling import binarize_image
from _utils_surface_sampling import cylindrical_projection, mesh_parameterization_heatmap, sample_fluorescence, cylindrical_projection_pca, unfold_to_int_map, unfold_surface
from PyQt5.QtWidgets import QApplication, QFileDialog

if __name__ == '__main__':

    app = QApplication([])
    fname = QFileDialog.getOpenFileName(caption='Select a the .csv file to use as input',filter='*.tif')[0]
    original_fname = QFileDialog.getOpenFileName(caption='Select a the .csv file to use as input',filter='*.h5')[0]
    # fname = '/Users/avillars/Desktop/Surface_sampling_project/test_data/converted_img/probabilities.tif'
    # original_fname = '/Users/avillars/Desktop/Surface_sampling_project/test_data/converted_img/test_h5.h5'

    # fname_list = glob('S:/lightsheet/iber/Alexis/Nick_conversions/*_seg*.h5')

    # with h5py.File(fname,'r') as f:
    #     label_image = f[list(f.keys())[0]][...]

    label_image = tifffile.imread(fname)
    print(label_image.shape)

    #rescale the image
    # scale_factor_z = 0.25
    # scale_factor_y = 0.25
    # scale_factor_x = 0.25
    # label_image_rescaled = rescale(label_image, (1, scale_factor_z, scale_factor_y, scale_factor_x), anti_aliasing=False)

    # binarise the image 
    closing_size =10
    label_image = binarize_image(im = label_image, threshold = 0.1, closing_size=closing_size)

    # the binarisation closes holes through dilation: erode the image to correct the dilation on the sides. 
    label_image = binary_erosion(label_image, footprint=cube(closing_size))

    # # load the original images to measure the pixel intensities 
    with h5py.File(original_fname,'r') as f:
        original_im = f['downscaled'][...]
    # original_im_rescaled = rescale(original_im, (1, scale_factor_z, scale_factor_y, scale_factor_x), anti_aliasing=False)


    mesh = binary_mask_to_surface(
        label_image,
        n_mesh_smoothing_interations=10
    )

    boundary_dilation_size = 1
    boundary_mask = make_boundary_mask(
            label_image=label_image, boundary_dilation_size=boundary_dilation_size
        )
    boundary_labels = label_image.copy()
    boundary_labels[np.logical_not(boundary_mask)] = 0

    # # surface_measurements = boundary_mask * original_im_rescaled

    # # Step 3: Extract surface points
    surface_voxels = np.argwhere(label_image == 1)  # Nx3 array of (z, y, x)
    z, y, x = surface_voxels[:, 0], surface_voxels[:, 1], surface_voxels[:, 2]

    # Sample fluorescence intensity at these points
    fig, ax = plt.subplots(1,4, figsize=(8, 6))
    for channel in range(0,4):
        print(channel)
        fluorescence_values = original_im[channel][z, y, x]
        # Normalize the fluorescence values
        f_min, f_max = fluorescence_values.min(), fluorescence_values.max()
        u, v = cylindrical_projection(surface_voxels, mesh, center_method='centroid')
        
        ax[channel].scatter(u, v, c=fluorescence_values, vmin=f_min, vmax=f_max, cmap="inferno")
        ax[channel].set_box_aspect(1)
        ax[channel].set_title(str(channel))
    fig.supxlabel("Longitude (u)")
    fig.supylabel("Latitude (v)")
    fig.suptitle("Cylindrical Projection of 3D Elliptical Surface")
    fig.tight_layout()
    plt.show()


    fig, ax = plt.subplots(1,4, figsize=(8, 6))
    for channel in range(0,4):
        vertices_2D, tri = mesh_parameterization_heatmap(mesh)
        fluorescence_values = sample_fluorescence(mesh, original_im[channel,...])
        f_min, f_max = fluorescence_values.min(), fluorescence_values.max()

        # Step 3: Plot the heatmap
        pcm = ax[channel].tricontourf(vertices_2D[:, 0], vertices_2D[:, 1], tri.simplices, fluorescence_values, cmap="inferno", vmin=f_min, vmax=f_max)
        # fig.colorbar(pcm, ax= ax[channel], label="Fluorescence Intensity")
        ax[channel].set_box_aspect(1)
    fig.suptitle("Flattened Mesh with Fluorescence Heatmap")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1,4, figsize=(8, 6))
    for channel in range(0,4):
        theta, h = cylindrical_projection_pca(mesh)
        fluorescence_values = sample_fluorescence(mesh, original_im[channel,...])
        # Step 3: Plot the heatmap
        pcm =ax[channel].tricontourf(theta, h, fluorescence_values, levels=100, cmap="inferno", vmin=f_min, vmax=f_max)
        # fig.colorbar(pcm, ax= ax[channel], label="Fluorescence Intensity")
        ax[channel].set_box_aspect(1)
    fig.supxlabel("Azimuthal Angle (θ)")
    fig.supylabel("Height (h)")
    fig.suptitle("PCA-Based Cylindrical Projection with Fluorescence Heatmap")
    plt.show()

    fig, ax = plt.subplots(1,4, figsize=(8, 6))
    for channel in range(0,4):
        theta, h = unfold_surface(mesh)
        fluorescence_values = sample_fluorescence(mesh, original_im[channel,...])
        # Step 3: Plot the heatmap
        pcm = ax[channel].tricontourf(theta, h, fluorescence_values, cmap="inferno", vmin=f_min, vmax=f_max)
        # fig.colorbar(pcm, ax= ax[channel], label="Fluorescence Intensity")
        ax[channel].set_box_aspect(1)
    fig.supxlabel("Azimuthal Angle (θ)")
    fig.supylabel("Height (h)")
    fig.suptitle("PCA-Based Cylindrical Projection with Fluorescence Heatmap")
    plt.show()

    fig, ax = plt.subplots(1,4, figsize=(8, 6))
    for channel in range(0,4):
        intensity_map = unfold_to_int_map(mesh, original_im[channel,...])
        ax[channel].imshow(intensity_map, cmap="inferno")
        # ax[channel].set_box_aspect(1)
    plt.show()
