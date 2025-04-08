import h5py
import numpy as np 
from skimage.exposure import rescale_intensity
from skimage.transform import rescale
import dask.array as da
from dask.diagnostics import ProgressBar

from PyQt5.QtWidgets import QApplication, QFileDialog
from multiprocessing import get_context
from functools import partial
from glob import glob
import os 

fname = 'S:/lightsheet/iber/Alexis/Nick_conversions/original_images/microfluidics 24h sag2 .ims'

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
    # p_low_val, p_high_val = np.percentile(im_rescaled, (p_low, p_high))
    # im_normalized = rescale_intensity(im_rescaled, in_range=(p_low_val, p_high_val))
    return im_rescaled

def downscale(im: np.ndarray):
    im_downscaled = rescale(im, (1, 0.25, 0.25, 0.25), anti_aliasing=False)
    return im_downscaled    

def read_and_stack(fname):

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
            # dask_array = da.from_array(dataset, chunks="auto")  # Automatically choose chunk sizes

            # Store the lazy Dask array
            lazy_channels.append(dataset)
        lazy_channels = np.stack(lazy_channels)
        # Stack the channels lazily
        # stacked_data = da.stack(lazy_channels, axis=0)
    return lazy_channels, pixel_x, pixel_y, pixel_z

def convert_image(fname, output_directory):
    print('starts reading')
    lazy_channels, pixel_x, pixel_y, pixel_z = read_and_stack(fname)
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

    im_rescaled_normalised = rescale_and_normalize(lazy_channels, scale_factor_x, scale_factor_y, scale_factor_z)

    im_downscaled = downscale(im_rescaled_normalised)

    output_fname = os.path.basename(fname)
    output_fpath = os.path.join(output_directory, output_fname.replace('.ims','.h5'))

    with h5py.File(output_fpath, 'w') as f_out:
        dset_normed = f_out.create_dataset(
            'raw_rescaled',
            im_rescaled_normalised.shape,
            dtype=im_rescaled_normalised.dtype,
            data=im_rescaled_normalised,
            compression='gzip'
        )

        dset_normed.attrs['original_filename'] = fname
        dset_normed.attrs['x_in_um'] = pixel_x
        dset_normed.attrs['x_out_um'] = pixel_z
        dset_normed.attrs['y_in_um'] = pixel_y
        dset_normed.attrs['y_out_um'] = pixel_z
        dset_normed.attrs['z_in_um'] = pixel_z
        dset_normed.attrs['z_out_um'] = pixel_z

        dset_rescaled = f_out.create_dataset(
            'downscaled',
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

if __name__ == '__main__':
    app = QApplication([])
    input_directory = QFileDialog.getExistingDirectory(caption='Select an input folder')
    output_directory = QFileDialog.getExistingDirectory(caption='Select an output folder')
    print("output folder : ", output_directory)
    print("input file : ", input_directory)

    flist = glob(input_directory+'/*.ims')
    n_images = len(flist)
    print(f'converting {n_images} images')
    with get_context("spawn").Pool(processes=2) as pool:
        pool.map(partial(convert_image, output_directory=output_directory), flist)
    print('Done')