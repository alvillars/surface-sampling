"""
Script to convert Imaris files to h5 used in the neural tube pipeline.

CHANNELS should be formatted:
	<combination number>: ['staining 0', 'olig channel', 'nkx channel', 'staining 3, optional']

My combination (MW added): <combination number>: ['laminin channel', 'shh channel', 'olig channel', 'nkx channel']

For example if you add combination 4 where laminin is Channel 1, olig2 is Channel 0,
nkx2.2 is Channel 2, and GFP is Channel 3, it would be:

CHANNELS = {
	2: ['Channel 0','Channel 2', 'Channel 1'],
	3: ['Channel 2', 'Channel 0', 'Channel 1'],
	4: ['Channel 1', 'Channel 0', 'Channel 2', 'Channel 3']
}

0. Update the CHANNELS, OUTPUT_DIRECTORY, and TABLE_PATH
1. Copy file to shared computer
2. Open anacaonda
3. activate the splineslicer environment (conda activate spline slicer)
4. Run the script: python path/to/convert_data_2021072.py
"""


import os

from multiprocessing import get_context
from functools import partial

import h5py
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
from skimage.transform import rescale
from PyQt5.QtWidgets import QApplication, QFileDialog

# channels should be ordered shh, olig2, nkx
#CHANNELS = {
#    4: ["Channel 0",  "Channel 2", "Channel 3", "Channel 1"]
#}
#OUTPUT_DIRECTORY = '' #= r"C:\\Users\marvwyss\Desktop\080223_splineslicer_EZ1"
#TABLE_PATH =''  #= r"C:\\Users\marvwyss\Desktop\080223_splineslicer_EZ1\sample_to_convert.csv"

def rescale_and_normalize(
		im: np.ndarray,
		scale_factor_x: float,
		scale_factor_y: float,
		scale_factor_z: float,
		p_low: float = 2,
		p_high: float = 98

)-> np.ndarray:
	im_rescaled = rescale(im, (scale_factor_z, scale_factor_y, scale_factor_x), anti_aliasing=False)

	p_low_val, p_high_val = np.percentile(im_rescaled, (p_low, p_high))
	im_normalized = rescale_intensity(im_rescaled, in_range=(p_low_val, p_high_val))

	im_downscaled = rescale(im, (0.25, 0.25, 0.25), anti_aliasing=False)
	return im_downscaled, im_normalized


def convert_image(row, output_directory):
	output_dir = output_directory
	#print('that dir', output_dir)

	fname = row['File']
	print(f'Image: {fname}')

	User_order = row['channel_order']
	channel_order = [i for i in map(int, User_order.split(','))]
	channel_names = ["Channel "+str(chan) for chan in channel_order]
	mapped_chan_names = [row[chan] for chan in channel_names]

	raw_images = []
	with h5py.File(fname, 'r') as f:
		for channel in channel_names:
			raw_images.append(
				 f['DataSet']['ResolutionLevel 0']['TimePoint 0'][channel]['Data'][:]
			 )

	print('image loaded')

	# from X/Y vs z resolution provided by laura
	scale_factor_x = row['x_in'] / row['x_out']
	scale_factor_y = row['y_in'] / row['y_out']
	scale_factor_z = row['z_in'] / row['z_out']

	print(scale_factor_x, scale_factor_y, scale_factor_z)	


	rescaled_and_normed_images = []
	for raw_image in raw_images:
		rescaled_and_normed_images.append(
			rescale_and_normalize(
				raw_image,
				scale_factor_x=scale_factor_x,
				scale_factor_y=scale_factor_y,
				scale_factor_z=scale_factor_z,
				p_low=2,
				p_high=99
			)
		)
	rescaled_images = [image[0] for image in rescaled_and_normed_images]
	im_rescaled = np.stack(rescaled_images)

	normalized_images = [image[1] for image in rescaled_and_normed_images]
	im_normalized = np.stack(normalized_images)

	
	print(im_normalized.shape)


	samples = row['Sample']
	output_fname = f'gastruloids_sample_{samples}.h5'
	output_fpath = os.path.join(output_dir, output_fname)
		
	with h5py.File(output_fpath, 'w') as f_out:
		dset_normed = f_out.create_dataset(
			'raw_rescaled_normalized',
			im_normalized.shape,
			dtype=im_normalized.dtype,
			data=im_normalized,
			compression='gzip'
		)
		dset_normed.attrs['original_filename'] = fname
		dset_normed.attrs['x_in_um'] = row['x_in']
		dset_normed.attrs['x_out_um'] = row['x_out']
		dset_normed.attrs['y_in_um'] = row['y_in']
		dset_normed.attrs['y_out_um'] = row['y_out']
		dset_normed.attrs['z_in_um'] = row['z_in']
		dset_normed.attrs['z_out_um'] = row['z_out']
		for i, chan in enumerate(mapped_chan_names):
			dset_normed.attrs['chan_'+str(i)] = chan
			

		dset_rescaled = f_out.create_dataset(
			'down_scaled',
			im_rescaled.shape,
			dtype=im_rescaled.dtype,
			data=im_rescaled,
			compression='gzip'
		)
		dset_rescaled.attrs['original_filename'] = fname
		dset_rescaled.attrs['x_in_um'] = row['x_in']
		dset_rescaled.attrs['x_out_um'] = row['x_out']*0.25
		dset_rescaled.attrs['y_in_um'] = row['y_in']
		dset_rescaled.attrs['y_out_um'] = row['y_out']*0.25
		dset_rescaled.attrs['z_in_um'] = row['z_in']
		dset_rescaled.attrs['z_out_um'] = row['z_out']*0.25
		for i, chan in enumerate(mapped_chan_names):
			dset_rescaled.attrs['chan_'+str(i)] = chan

	print(f'{output_fpath} saved')


if __name__ == '__main__':
	app = QApplication([])
	table_path = QFileDialog.getOpenFileName(caption='Select a the .csv file to use as input',filter='*.csv')[0]
	output_directory = QFileDialog.getExistingDirectory(caption='Select an output folder')
	print("output folder : ", output_directory)
	print("input file : ", table_path)

	df = pd.read_csv(table_path, sep=",")
	n_images = len(df)
	print(f'converting {n_images} images')
	print(df.head())
	rows = [r for _, r in df.iterrows()] 
	with get_context("spawn").Pool(processes=2) as pool:
		pool.map(partial(convert_image, output_directory=output_directory), rows)
	print('Done')