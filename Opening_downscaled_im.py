import napari
import h5py
from PyQt5.QtWidgets import QApplication, QFileDialog


if __name__ == '__main__':
    app = QApplication([])
    input_directory = QFileDialog.getOpenFileName(caption='Select a the .csv file to use as input',filter='*.h5')[0]

    fname = input_directory
    with h5py.File(fname, 'r') as f: 
        im = f['downscaled'][...]

    viewer = napari.Viewer()
    viewer.add_image(im, visible=False)

    napari.run()