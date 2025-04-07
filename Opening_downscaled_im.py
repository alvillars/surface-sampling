import napari
import h5py

fname = '/Users/avillars/Desktop/Surface_sampling_project/test_data/converted_img/test_h5.h5'
with h5py.File(fname, 'r') as f: 
    im = f['downscaled'][...]

viewer = napari.Viewer()
viewer.add_image(im, visible=False)

napari.run()