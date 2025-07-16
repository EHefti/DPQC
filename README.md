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
3. Run the code


## Description
1. Pre-assessment: Identify wells that have enough activity to sort them
2. Spike Sorting: Use Kilosort (or another sorter) to sort the raw data
3. Quality Control: Label one well manually to train the model, then apply the model to label the rest of your data



## Phy for Manual Labeling
There is an alternative to label your units manually, which displays more information in the GUI. The platform is called Phy, but you need to install phy separately (optional):

Run: conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets
Run: conda activate phy2
Install phy: pip install git+https://github.com/cortex-lab/phy.git

The documentation to explain the GUI can be found here `https://phy.readthedocs.io/en/latest/`


You can also start the GUI from python, but you will have to change the kernel which makes you loose your cached variables. Use the code below:

```python
from phy.apps.template import template_gui
from pathlib import Path

save_root = 'your/path/to/your/sorted/wells/'
well_id = 'well010'
params_path = Path(save_root) / well_id / 'sorter_output' / 'params.py'
template_gui(params_path)
```
