from _utils_surface_sampling import create_ellipse, make_xy_gradient
from _utils_surface_sampling import cylindrical_projection, mesh_parameterization_heatmap, sample_fluorescence, cylindrical_projection_pca, unfold_surface, unfold_to_int_map
from morphometrics.utils.surface_utils import binary_mask_to_surface
from morphometrics.utils.image_utils import make_boundary_mask
import numpy as np 
import matplotlib.pyplot as plt 
import napari 

if __name__ == '__main__':
    label_image, ellipsoid_mask = create_ellipse()
    x_gradient = make_xy_gradient(k=5, label_image=label_image)
    # distance_map = make_inside_gradient(ellipsoid_mask)

    # Combine gradients (e.g., multiply for smooth transition)
    combined_intensity = x_gradient  * label_image.astype(np.float32)
    # Note: Multiplying by label_image ensures that the gradients only affect the object.

    # mesh the ellipsoid
    mesh = binary_mask_to_surface(
        label_image,
        n_mesh_smoothing_iterations=10
    )
    
    boundary_dilation_size = 1
    boundary_mask = make_boundary_mask(
            label_image=label_image, boundary_dilation_size=boundary_dilation_size
        )
    boundary_labels = label_image.copy()
    boundary_labels[np.logical_not(boundary_mask)] = 0

    surface_measurements = boundary_mask * combined_intensity

    # Step 3: Extract surface points
    surface_voxels = np.argwhere(label_image == 1)  # Nx3 array of (z, y, x)
    z, y, x = surface_voxels[:, 0], surface_voxels[:, 1], surface_voxels[:, 2]

    # Sample fluorescence intensity at these points
    fluorescence_values = combined_intensity[z, y, x]
    # Normalize the fluorescence values
    f_min, f_max = fluorescence_values.min(), fluorescence_values.max()

    u, v = cylindrical_projection(surface_voxels, mesh, center_method='centroid')

    plt.figure(figsize=(8, 6))
    plt.scatter(u, v, c=fluorescence_values, vmin=f_min, vmax=f_max, cmap="inferno")
    plt.xlabel("Longitude (u)")
    plt.ylabel("Latitude (v)")
    plt.title("Cylindrical Projection of 3D Elliptical Surface")
    plt.show()

    vertices_2D, tri = mesh_parameterization_heatmap(mesh)
    fluorescence_values = sample_fluorescence(mesh, combined_intensity)

    # Step 3: Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.tricontourf(vertices_2D[:, 0], vertices_2D[:, 1], tri.simplices, fluorescence_values, cmap="inferno", vmin=f_min, vmax=f_max)
    plt.colorbar(label="Fluorescence Intensity")
    plt.title("Flattened Mesh with Fluorescence Heatmap")
    plt.axis("equal")
    plt.show()

    theta, h = cylindrical_projection_pca(mesh)

    # Step 3: Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.tricontourf(theta, h, fluorescence_values, cmap="inferno", vmin=f_min, vmax=f_max)
    plt.colorbar(label="Fluorescence Intensity")
    plt.xlabel("Azimuthal Angle (θ)")
    plt.ylabel("Height (h)")
    plt.title("PCA-Based Cylindrical Projection with Fluorescence Heatmap")
    plt.show()


    theta, h = unfold_surface(mesh)

    # Step 3: Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.tricontourf(theta, h, fluorescence_values, cmap="inferno", vmin=f_min, vmax=f_max)
    plt.colorbar(label="Fluorescence Intensity")
    plt.xlabel("Azimuthal Angle (θ)")
    plt.ylabel("Height (h)")
    plt.title("PCA-Based Cylindrical Projection with Fluorescence Heatmap")
    plt.show()

    intensity_map = unfold_to_int_map(mesh, combined_intensity)
        
    plt.figure(figsize=(8, 6))
    plt.imshow(intensity_map, cmap="inferno")
    plt.show()

    # viewer = napari.Viewer(ndisplay=3)

    # # add the image to the viewer
    # viewer.add_labels(
    #     label_image,
    #     visible=False
    # )


    # viewer.add_image(
    #     combined_intensity,
    #     rendering="mip",
    #     iso_threshold=0,
    #     visible=False
    # )

    # viewer.add_image(
    #     surface_measurements,
    #     rendering="mip",
    #     iso_threshold=0,
    #     visible=False
    # )

    # viewer.add_surface(
    #     (mesh.vertices, mesh.faces, np.ones((len(mesh.vertices),)))
    # )

    # napari.run()