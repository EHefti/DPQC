import os
import numpy as np
import pandas as pd
import bombcell as bc
import shutil

def writeNPY(data, filename):
    np.save(filename, data)

def bombcell_to_phy(bc_savePath: list[str], overwrite: bool = False):
    """
    Function to load bombcell output from bc_savePath and apply it to the
    original phy output.

    Args:
        bc_savePath (list[str]): Directories of bombcell output.
        overwrite (bool, optional): Flag to indicate if output should be overwritten. Defaults to False.
    """

    for save_path in bc_savePath:
        if not os.path.exists(os.path.join(save_path, "templates._bc_qMetrics.parquet")):
            continue

        param, qMetric, fraction_RPVs = bc.load_bc_results(save_path)
        qc_path = os.path.join(param['ephysKilosortPath'], 'qc_output')

        if not os.path.exists(os.path.join(qc_path, 'phy_ids.npy')) or overwrite:
            # Load bombcell output
            unitType = bc.get_quality_unit_type(param, qMetric, save_path)

            # Load spike sorting results
            spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, _, _ = \
                bc.load_ephys_data(param['ephysKilosortPath'], save_path)

            # Load KS Labels
            ks_labels = pd.read_csv(os.path.join(param['ephysKilosortPath'], "cluster_KSLabel.tsv"), sep='\t')

            unique_units = np.unique(spikeTemplates)
            
            # Adjusting for 0-based indexing in Python for cluster_id
            ks_good = (ks_labels['KSLabel'][ks_labels['cluster_id'].isin(unique_units)] == 'g').to_numpy()
            
            good_units = unique_units[(unitType == 1) & (ks_good == 1)]
            good_spikes = np.isin(spikeTemplates, good_units)

            # Create a mapping from original good_units to new 0-based indices
            good_unit_mapping = {unit: i for i, unit in enumerate(good_units)}
            goodSpikeTemplates = np.array([good_unit_mapping[t] for t in spikeTemplates[good_spikes]])

            goodSpikeTimes = spikeTimes_samples[good_spikes]
            goodTemplateMatrix = templateWaveforms[good_units, :, :] # Assuming templateWaveforms indexed by unit ID

            if not os.path.exists(qc_path):
                os.makedirs(qc_path)

            params_file = os.path.join(param['ephysKilosortPath'], 'params.py')
            coor_file = os.path.join(param['ephysKilosortPath'], 'channel_positions.npy')

            writeNPY(goodSpikeTemplates, os.path.join(qc_path, 'spike_templates.npy'))
            writeNPY(goodSpikeTimes, os.path.join(qc_path, 'spike_times.npy'))
            writeNPY(goodTemplateMatrix, os.path.join(qc_path, 'templates.npy'))
            writeNPY(good_units, os.path.join(qc_path, 'phy_ids.npy'))

            shutil.copyfile(params_file, os.path.join(qc_path, 'params.py'))
            shutil.copyfile(coor_file, os.path.join(qc_path, 'channel_positions.npy'))