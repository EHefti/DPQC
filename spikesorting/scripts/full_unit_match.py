"""
Script Name: full_unit_match.py
Description: This script matches units based on their position across different sortings 
Author: Philipp Hornauer
Date: 2024-01-06
"""

import os,sys
import warnings

import spikeinterface.full as si
import numpy as np

from tqdm import tqdm
from pathlib import Path

sys.path.append("/home/phornauer/Git/spikesorting/")
from spikesorting import unit_match as um

warnings.filterwarnings("ignore", category=RuntimeWarning)
si.set_global_job_kwargs(n_jobs=64, progress_bar=False)

parent_path = '/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Maxtwo/phornauer/'
project_name = 'Torsten_2'
recording_date = '241114'
chip_id = 'T002523'
assay_name = 'Network'
assay_id = '*'

param = {
    'spike_width': 30, # in samples
    'waveidx': np.arange(5,15), # index of the waveform to be used for the metric
    'peak_loc': 10, # index of the peak location in the waveform
    'no_shanks': 1, # unit_match parameter for neuropixel probe
    'shank_dist': 0, # unit_match parameter for neuropixel probe
}

sel_idx_list = [np.r_[0,3:12],np.r_[0,1],np.r_[3,2]] # Order in which the units are matched

for w in tqdm(range(13,17)):
    well_id = f'well{w:03d}'
    path_parts = [parent_path, project_name, recording_date, chip_id, assay_name, assay_id, well_id, "sorter_output","spike_times.npy"]
    path_list = um.get_sorting_path_list(path_parts)
    
    if len(path_list) < 10:
        continue
    
    parts = Path(path_list[0]).parts
    save_path = Path(os.path.join(*parts[:-3], "UM_data", parts[-2]))
    param['save_path'] = save_path
    # Generate save path from the first sorting path
        
    waveforms, channel_pos = um.generate_templates(path_list)
    
    score_matrix, clus_info, param = um.get_score_matrix(waveforms, channel_pos, param, load_if_exists=True)
    
    score_container = um.make_score_container(score_matrix, clus_info['session_switch'])
    
    full_paths = um.kcl_match(score_container, sel_idx_list)
    
    um.save_matched_sortings(path_list, full_paths)
