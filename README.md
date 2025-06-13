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

   `conda create -n your_env_name python=3.9 matplotlib pathlib h5py numpy pandas tqdm scikit-learn spikeinterface bombcell -c conda-forge`

   _(this may take a while, don't kill the process)_

4. Activate the conda environment
5. Run the code
