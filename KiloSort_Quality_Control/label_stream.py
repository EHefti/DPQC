from pathlib import Path
import h5py
import spikeinterface.full as si 
import pandas as pd
from statistics import mean

def auto_label_stream(rec_path, save_root, stream_id, model_folder):
    """
    Function to label a stream using the trained model. 
    Labels are saved in the sorting output directory.
    
    Parameters:
    - rec_path (str): Path to the recording file.
    - save_root (str): Root directory where the sorting output is saved.
    - stream_id (str): Identifier for the stream to be labeled.
    - model_folder (str): Folder where the trained model is stored.
    
    Returns:
    - None
    """
    
    print("")
    print("------------------------")
    print("")
    print(f'Analyzing and Labeling {stream_id}:')
    h5 = h5py.File(rec_path)
    rec_name = list(h5['wells'][stream_id].keys())[0]
    rec = si.MaxwellRecordingExtractor(rec_path, stream_id=stream_id, rec_name=rec_name)

    path_to_sorting = Path(save_root) / stream_id / 'sorter_output'
    sorting = si.read_kilosort(folder_path=path_to_sorting)
    

    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec)
    analyzer.compute(['noise_levels','random_spikes','waveforms','templates','spike_locations','spike_amplitudes',
                      'correlograms','principal_components', 'quality_metrics', 'template_metrics'])
    
    labels_and_probabilities = si.auto_label_units(
        sorting_analyzer=analyzer,
        model_folder=model_folder,
        trust_model=True
    )

    # print how many units were labeled as good and bad
    good_units = sum(label == 'good' for label in labels_and_probabilities["label"])
    bad_units = sum(label == 'bad' for label in labels_and_probabilities["label"])
    print(f'Number of good units: {good_units}')
    print(f'Number of bad units: {bad_units}')

    avg_confidence = mean(labels_and_probabilities["probability"])
    print(f'Average confidence of the model: {avg_confidence:.3f}')
    print("")

    labels_and_probabilities = pd.DataFrame(labels_and_probabilities)
    labels_and_probabilities.to_csv(path_to_sorting / 'labels_and_probabilities.csv')