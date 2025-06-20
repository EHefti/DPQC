# DPQC
Data Processing &amp; Quality Control (DPQC) of electrophysiological data on microelectrode arrays.

DPQC includes the following steps:

1. Pre-Assessment
2. Sorting
3. Quality control
4. Post Processing

Given the type of the recording, one can continue with Axon_Tracking or DeePhys for the feature extraction.



## Set Up
1. Create a conda environment with the following command:

   `conda create -n your_env_name python=3.9 matplotlib pathlib h5py numpy pandas tqdm skops scikit-learn spikeinterface ipykernel -c conda-forge`

   _(change `your_env_name` to the name you want for your environement)_

2. Activate the conda environment
3. Install Bombcell: `pip install bombcell`
4. Run the code



## Phy for Manual Labeling
There is an alternative to label your units manually, which gives you more information in the GUI. The platform is called Phy, but you need to install phy separately (optional):

Run: conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets
Run: conda activate phy2
Install phy: pip install git+https://github.com/cortex-lab/phy.git
