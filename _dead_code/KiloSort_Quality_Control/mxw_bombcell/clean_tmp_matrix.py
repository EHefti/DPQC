import numpy as np

def clean_tmp_matrix(templateWaveforms: np.ndarray, 
                     spikeTimes_samples: np.ndarray, 
                     spikeTemplates: np.ndarray, 
                     templateAmplitudes: np.ndarray, 
                     pcFeatures: np.ndarray):
    nan_wf = np.where(np.all(np.isnan(templateWaveforms), axis=(1, 2)))[0]
    rm_spk_idx = np.isin(spikeTemplates, nan_wf)
    spikeTimes_samples = spikeTimes_samples[~rm_spk_idx]
    spikeTemplates = spikeTemplates[~rm_spk_idx]
    templateAmplitudes = templateAmplitudes[~rm_spk_idx]
    pcFeatures = pcFeatures[~rm_spk_idx, :, :]
    return spikeTimes_samples, spikeTemplates, templateAmplitudes, pcFeatures