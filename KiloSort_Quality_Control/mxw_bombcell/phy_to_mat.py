import os
import numpy as np
import scipy.io as sio

def phy_to_mat(phy_path_list: list[str]):
    for phy_path in phy_path_list:
        time_path = os.path.join(phy_path, 'spike_times.npy')
        template_path = os.path.join(phy_path, 'spike_templates.npy')
        
        if not (os.path.exists(time_path) and os.path.exists(template_path)):
            raise AssertionError(f"Required files not found in {phy_path}: 'spike_times.npy' and/or 'spike_templates.npy'")
        
        spike_times = np.load(time_path)
        spike_templates = np.load(template_path)
        
        save_path = os.path.join(phy_path, 'spikes.mat')
        
        spikes = {
            'times': spike_times,
            'unit': spike_templates
        }
        
        sio.savemat(save_path, {'spikes': spikes})