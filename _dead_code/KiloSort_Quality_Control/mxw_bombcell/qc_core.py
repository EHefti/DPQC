import os
import numpy as np
import bombcell as bc

from clean_tmp_matrix import clean_tmp_matrix
from bombcell_to_phy import bombcell_to_phy

def qc_core(sorting_path_list: list[str], param: dict):
    for sorting_path in sorting_path_list:
        if os.path.exists(os.path.join(sorting_path, 'templates.npy')):
            param['ephysKilosortPath'] = sorting_path
            savePath = os.path.join(sorting_path, 'bc_output')

            spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, \
            pcFeatureIdx, channelPositions = bc.load_ephys_data(sorting_path)

            spikeTimes_samples, spikeTemplates, templateAmplitudes, pcFeatures = \
                clean_tmp_matrix(templateWaveforms, spikeTimes_samples, spikeTemplates, templateAmplitudes, pcFeatures)
            
            qMetric, runtimes = bc.get_all_quality_metrics(param, spikeTimes_samples, spikeTemplates, 
                                                             templateWaveforms, templateAmplitudes, pcFeatures, 
                                                             pcFeatureIdx, channelPositions, savePath)
            
            unitType, unitTypeString = bc.get_quality_unit_type(param, qMetric)

            print(f"Found {np.sum(unitType == 1)} good single units")
                
    bc_path_list = [os.path.join(sp, "bc_output") for sp in sorting_path_list]

    overwrite = True
    bombcell_to_phy(bc_path_list, overwrite)