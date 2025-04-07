import h5py
import numpy as np 
from skimage.exposure import rescale_intensity
from skimage.transform import rescale
import dask.array as da
import time
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler

fname = '/Volumes/bs-dfs/lightsheet/iber/Alexis/Nick_conversions/original_images/microfluidics 24h sag2 .ims'

channel_order = [0, 1, 2, 3]
channel_names = ["Channel "+str(chan) for chan in channel_order]

def clean_string(input_string: str):
    cleaned_string = ''.join([item.decode('utf-8') for item in input_string])
    return cleaned_string

def explore_group(group, path="/"):
    for key in group.keys():
        item = group[key]
        item_path = f"{path}{key}"
        print(f"Path: {item_path}")
        if isinstance(item, h5py.Group):
            print("  Type: Group")
            print("  Attributes:")
            for attr_name, attr_value in item.attrs.items():
                print(f"    {attr_name}: {clean_string(attr_value)}")
            explore_group(item, path=f"{item_path}/")
        elif isinstance(item, h5py.Dataset):
            print("  Type: Dataset")
            print("  Shape:", item.shape)
            print("  Data type:", item.dtype)
            print("  Attributes:")
            for attr_name, attr_value in item.attrs.items():
                print(f"    {attr_name}: {attr_value}")

def rescale_and_normalize(im: np.ndarray,scale_factor_x: float,scale_factor_y: float, scale_factor_z: float,
		p_low: float = 2,
		p_high: float = 98):
     
    im_rescaled = rescale(im, (1, scale_factor_z, scale_factor_y, scale_factor_x), anti_aliasing=False)
    p_low_val, p_high_val = np.percentile(im_rescaled, (p_low, p_high))
    im_normalized = rescale_intensity(im_rescaled, in_range=(p_low_val, p_high_val))
    return im_normalized

def downscale(im: np.ndarray):
    im_downscaled = rescale(im, (1, 0.25, 0.25, 0.25), anti_aliasing=False)
    return im_downscaled    


raw_images = []
with h5py.File(fname, 'r') as f:
    pixel_x = f['DataSetInfo']['ZeissAttrs'].attrs.__getitem__(
        'ImageDocument/Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/ScalingX')
    pixel_y = f['DataSetInfo']['ZeissAttrs'].attrs.__getitem__(
        'ImageDocument/Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/ScalingY')
    pixel_z = f['DataSetInfo']['ZeissAttrs'].attrs.__getitem__(
        'ImageDocument/Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/ScalingZ')

    lazy_channels = []

    # Iterate over channels and create Dask arrays without loading data into memory
    for chan in channel_order:
        channel_name = channel_names[chan]

        # Access the dataset lazily using dask.array.from_array
        dataset = f['DataSet']['ResolutionLevel 0']['TimePoint 0'][channel_name]['Data']

        # Convert to Dask array (lazy loading, doesn't load into memory yet)
        dask_array = da.from_array(dataset, chunks="auto")  # Automatically choose chunk sizes

        # Store the lazy Dask array
        lazy_channels.append(dask_array)

    # Stack the channels lazily
    stacked_data = da.stack(lazy_channels, axis=0)

stacked_data = stacked_data.rechunk((1,288,1920,1920))
# Print shape (this won't trigger computation)
print("Lazy stacked shape:", stacked_data.shape)

# pixel values in um (multiply m by *1e6 to get um from attributes)
# this uses the clean_string function to filter all the b' weird strings formating in imaris
pixel_x = float(clean_string(pixel_x))*1e6
pixel_y = float(clean_string(pixel_y))*1e6
pixel_z = float(clean_string(pixel_z))*1e6

# get scaling factors to scale everything to the z scale for isotropic image
scale_factor_x = pixel_x / pixel_z
scale_factor_y = pixel_y / pixel_z
scale_factor_z = pixel_z / pixel_z
print(scale_factor_x, scale_factor_y, scale_factor_z)

im_rescaled_normalised = da.map_blocks(
    rescale_and_normalize,
    stacked_data,  # Dask array input
    scale_factor_x, scale_factor_y, scale_factor_z,  # Additional arguments
    dtype=stacked_data.dtype,
    chunks=(1, 288, int(1920//scale_factor_x), int(1920//scale_factor_y))
)

im_downscaled = da.map_blocks(
    downscale,
    im_rescaled_normalised,
    dtype=stacked_data.dtype,
    chunks=(1, 288//4, int(1920//scale_factor_x)//4, int(1920//scale_factor_y)//4) # Expected output dtypes
)

# ---- 🔹 Compute Lazily Only When Needed ----
print("Lazy Downscaled Shape:", im_downscaled.shape)

start_time = time.time()
output_fpath = '/Users/avillars/Desktop/Surface_sampling_project/test_h5.h5'
with ProgressBar():
    with h5py.File(output_fpath, 'w') as f_out:
        dset_normed = f_out.create_dataset(
            'raw_rescaled_normalized',
            im_rescaled_normalised.shape,
            dtype=im_rescaled_normalised.dtype,
            data=im_rescaled_normalised,
            compression='gzip'
        )
        da.compute(da.store(im_rescaled_normalised, dset_normed))    
        

        dset_normed.attrs['original_filename'] = fname
        dset_normed.attrs['x_in_um'] = pixel_x
        dset_normed.attrs['x_out_um'] = pixel_z
        dset_normed.attrs['y_in_um'] = pixel_y
        dset_normed.attrs['y_out_um'] = pixel_z
        dset_normed.attrs['z_in_um'] = pixel_z
        dset_normed.attrs['z_out_um'] = pixel_z

        dset_rescaled = f_out.create_dataset(
            'down_scaled',
            im_downscaled.shape,
            dtype=im_downscaled.dtype,
            data=im_downscaled,
            compression='gzip'
        )
        dset_rescaled.attrs['original_filename'] = fname
        dset_rescaled.attrs['x_in_um'] = pixel_x
        dset_rescaled.attrs['x_out_um'] = pixel_z*0.25
        dset_rescaled.attrs['y_in_um'] = pixel_y
        dset_rescaled.attrs['y_out_um'] = pixel_z*0.25
        dset_rescaled.attrs['z_in_um'] = pixel_z
        dset_rescaled.attrs['z_out_um'] = pixel_z*0.25
        da.store(im_downscaled, dset_rescaled)


end_time = time.time()  # End timer
print(f"Execution time: {end_time - start_time:.2f} seconds")