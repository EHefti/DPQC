# DPQC
Data Processing &amp; Quality Control (DPQC) of electrophysiological data on microelectrode arrays.

DPQC includes the following steps:

1. Pre-Assessment
2. Sorting
3. Quality control
4. Post Processing

Given the type of the recording, one can continue with Axon_Tracking or DeePhys for the feature extraction.


Command to create conda environment:
```python
conda create -n my_matlab_env python=3.10 numpy spikeinterface h5py matplotlib tqdm pandas seaborn -c conda-forge
```
