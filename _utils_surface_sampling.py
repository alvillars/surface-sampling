import numpy as np
from skimage.draw import ellipsoid
from scipy.ndimage import distance_transform_edt

from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

from scipy.ndimage import map_coordinates
from typing import Optional
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.morphology import binary_closing, cube


def cylindrical_projection(surface_voxels, mesh, center_method:str='centroid'):
    """
    Projects 3D points onto a 2D plane using cylindrical coordinates.
    
    Args:
        points (np.ndarray): Nx3 array of (x, y, z) coordinates.
    
    Returns:
        np.ndarray: Nx2 array of (u, v) coordinates.
    """
    z, y, x = surface_voxels[:, 0], surface_voxels[:, 1], surface_voxels[:, 2]

    # Normalize center (since ellipsoid is placed in a bigger volume)

    if center_method == 'centroid':
        center = mesh.vertices.mean(axis=0)
    elif center_method == 'center_of_mass':
        center = mesh.center_mass
    elif center_method == 'bounding_box':
        center = mesh.bounding_box.centroid
    else:
        raise ValueError("Invalid center_method. Choose from 'centroid', 'center_of_mass', or 'bounding_box'.")

    x = x - center[2]
    y = y - center[1]
    z = z - center[0]

    # Step 3: Convert to Cylindrical (Longitude-Latitude) Projection
    theta = np.arctan2(y, x)  # Longitude (Azimuthal angle)
    phi = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))  # Latitude (Elevation angle)

    # Normalize to [0,1] for visualization
    u = (theta + np.pi) / (2 * np.pi)  # Scale longitude to [0,1]
    v = (phi + np.pi/2) / np.pi  # Scale latitude to [0,1]
    
    return u, v

def create_ellipse(size_im=[100, 100, 100], size_ellipse=[40, 20, 20]):

    # Create the full image and add the ellipsoid mask.
    label_image = np.zeros((size_im[0], size_im[1], size_im[2]), dtype=np.uint16)
    # Create an ellipsoid mask. (The ellipsoid is centered in its own volume.)
    ellipsoid_mask = ellipsoid(size_ellipse[0], size_ellipse[1], size_ellipse[2])

    mask_shape = np.array(ellipsoid_mask.shape)  # (D, H, W)
    D, H, W = mask_shape  # Depth, Height, Width

    # Compute placement dynamically
    center = np.array(label_image.shape) // 2  # Center of label_image
    start = center - mask_shape // 2  # Compute start indices
    end = start + mask_shape  # Compute end indices


    # Place the ellipsoid inside the larger image.
    label_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = ellipsoid_mask

    return label_image, ellipsoid_mask

def make_xy_gradient(k, label_image):
    # Create the left-to-right gradient.
    # We assume the x-axis (axis=0) is left-to-right.
    # Normalize x-coordinates between 0 and 1.
    x = np.linspace(0, 1, label_image.shape[0])
    # Create a 3D gradient image by broadcasting x along the y and z axes.
    x = np.linspace(0, 1, label_image.shape[0])  # Normalize x from 0 to 1
    x_gradient = np.exp(-k * x)[:, np.newaxis, np.newaxis]  # Shape: (100,1,1), allows broadcasting
    return x_gradient

def make_inside_gradient(ellipsoid_mask, label_image): 
    # Distance transform for inside-to-outside gradient
    distance_inside = distance_transform_edt(ellipsoid_mask)
    distance_inside_norm = distance_inside / np.max(distance_inside)  # Normalize from 0 to 1

    mask_shape = np.array(ellipsoid_mask.shape)  # (D, H, W)

    # Compute placement dynamically
    center = np.array(label_image.shape) // 2  # Center of label_image
    start = center - mask_shape // 2  # Start indices
    end = start + mask_shape  # End indices

    # Create an empty distance map
    distance_map = np.zeros_like(label_image, dtype=np.float32)

    # Place the distance transform inside label_image
    distance_map[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = distance_inside_norm

    return distance_map

def mesh_parameterization_heatmap(mesh):
    """
    Projects a 3D mesh onto a 2D plane using UV parameterization and overlays fluorescence data.

    Args:
        mesh (trimesh.Trimesh): The input 3D mesh.
    Returns:
        None (Displays the heatmap)
    """
    
    # Step 1: Compute UV parameterization (flattening)
    vertices_3D = mesh.vertices  # (N, 3) original 3D coordinates
    faces = mesh.faces  # (M, 3) triangle faces

    # Use PCA to find a natural 2D projection (or use LSCM/ABF parameterization if needed)

    pca = PCA(n_components=2)
    vertices_2D = pca.fit_transform(vertices_3D)  # Project to best 2D plane

    # Step 2: Create a triangulation in 2D
    return vertices_2D, Delaunay(vertices_2D)

def sample_fluorescence(mesh, fluorescence_matrix):
    """
    Samples fluorescence values at mesh vertices using trilinear interpolation.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh.
        fluorescence_matrix (np.ndarray): 3D fluorescence volume.

    Returns:
        np.ndarray: Fluorescence values sampled at mesh vertices.
    """
    # Mesh vertices (X, Y, Z) as floating point coordinates
    vertices = mesh.vertices  

    # Get shape of fluorescence matrix
    shape = fluorescence_matrix.shape

    # Normalize vertex coordinates to fluorescence matrix indices
    coords = np.array([
        vertices[:, 0] * (shape[0] / mesh.bounds[1, 0]),  # Scale X
        vertices[:, 1] * (shape[1] / mesh.bounds[1, 1]),  # Scale Y
        vertices[:, 2] * (shape[2] / mesh.bounds[1, 2])   # Scale Z
    ])

    # Sample fluorescence values using trilinear interpolation
    sampled_fluorescence = map_coordinates(fluorescence_matrix, coords, order=1, mode='nearest')

    return sampled_fluorescence

def cylindrical_projection_pca(mesh):
    """
    Projects a 3D mesh onto a 2D cylindrical space using PCA to define the main axis.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh.
        fluorescence_matrix (np.ndarray): 3D fluorescence volume.
        
    Returns:
        None (Displays the cylindrical projection heatmap)
    """
    # Compute PCA to find the main axis
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    main_axis = pca.components_[0]  # First principal component

    # Project vertices onto the main axis to compute height (h)
    h = np.dot(mesh.vertices, main_axis)

    # Compute the radial components (x', y') in the plane perpendicular to main_axis
    perp_basis_1 = pca.components_[1]  # Second principal component (perpendicular)
    perp_basis_2 = pca.components_[2]  # Third principal component (perpendicular)
    
    x_perp = np.dot(mesh.vertices, perp_basis_1)
    y_perp = np.dot(mesh.vertices, perp_basis_2)

    # Compute cylindrical coordinates
    r = np.sqrt(x_perp**2 + y_perp**2)  # Radius
    theta = np.arctan2(y_perp, x_perp)  # Azimuthal angle (radians)

    # Normalize theta between 0 and 2π
    theta = (theta - np.min(theta)) / (np.max(theta) - np.min(theta)) * (2 * np.pi)

    return theta, h

def unfold_surface(mesh): 
    # Step 4: Compute the principal axis using PCA (for cylindrical projection)
    pca = PCA(n_components=1)
    axis = pca.fit(mesh.vertices).components_[0]
    proj = mesh.vertices @ axis  # Projection of points onto the axis (height)

    # Compute cylindrical coordinates: radius and angle (phi)
    centered = mesh.vertices - np.mean(mesh.vertices, axis=0)  # Center the object
    radii = np.linalg.norm(centered[:, :2], axis=1)  # Radius in XY plane
    phi = np.arctan2(centered[:, 1], centered[:, 0])  # Azimuthal angle (phi)

    # Normalize the coordinates for unwrapping
    phi_normalized = (phi + np.pi) / (2 * np.pi)  # Map phi to [0, 1]
    height_normalized = (proj - proj.min()) / (proj.max() - proj.min())  # Map height to [0, 1]

    return phi_normalized, height_normalized

def unfold_to_int_map(mesh, combined_intensity): 
    # Step 3: Convert Cartesian coordinates to cylindrical coordinates
    x, y, z = mesh.vertices.T
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Step 4: Bin the intensity data by spatial regions
    bins_theta = np.linspace(-np.pi, np.pi, num=100)  # 10-degree bins
    bins_z = np.linspace(np.min(z), np.max(z), num=36)  # Height bins

    # Digitize the data into bins
    theta_indices = np.digitize(theta, bins_theta)
    z_indices = np.digitize(z, bins_z)

    fluorescence_values = map_coordinates(combined_intensity, mesh.vertices.T, order=1)
    # Compute mean intensity per bin
    intensity_map = np.zeros((len(bins_theta) - 1, len(bins_z) - 1))
    for i in range(1, len(bins_theta)):
        for j in range(1, len(bins_z)):
            mask = (theta_indices == i) & (z_indices == j)
            if np.any(mask):
                intensity_map[i - 1, j - 1] = np.mean(fluorescence_values[mask])
    
    return intensity_map

def keep_largest_region(binary_im: np.ndarray) -> np.ndarray:
    """Keep only the largest region in a binary image. The region
    is calculated using the scipy.ndimage.label function.

    Parameters
    ----------
    binary_im : np.ndarray
        The binary image to filter for the largest region.

    Returns
    -------
    binarized_clean : np.ndarray
        The binary_im with only the largest conencted region.
    """
    label_im = ndi.label(binary_im)[0]
    rp = regionprops_table(label_im, properties=["label", "area"])
    if len(rp["area"]) > 0:
        max_ind = np.argmax(rp["area"])
        max_label = rp["label"][max_ind]
        binarized_clean = np.zeros_like(label_im, dtype=bool)
        binarized_clean[label_im == max_label] = True
    else:
        binarized_clean = np.zeros_like(label_im, dtype=bool)

    return binarized_clean

def binarize_image(im: np.ndarray, threshold: float = 0.5, closing_size: Optional[int] = None) -> np.ndarray:
    """Binaraize an image and keep only the largest structure. Small holes are filled
    with the scipy.ndimage.binary_fill_holes() function.

    Parameters
    ----------
    im : np.ndarray
        The image to be binarized. The image is expected to be ordered (c, z, y, x).
    channel : int
        The channel to binarize in im. The image is expected to be ordered (c, z, y, x).
    threshold : float
        Threshold for binarization. Values less than or equal to this
        value will be set to False and values greater than this value
        will be set to True.
    close_size: Optional[int]
        The size of morphological closing to apply to the binarized image.
        This is generally to fill in holes. If None, no closing will be applied.
        The default value is None.

    Returns
    -------
    binarized_clean : np.ndarray
        The binarized image containing only the largets segmented
        region.
    """
    # make the mask and label image
    selected_channel = im
    mask_im = selected_channel > threshold
    mask_im_filled = ndi.binary_fill_holes(mask_im)

    # keep only the largest structure
    binarized_clean = keep_largest_region(mask_im_filled)

    if closing_size is not None:
        return binary_closing(binarized_clean, footprint=cube(closing_size))
    else:
        return binarized_clean
