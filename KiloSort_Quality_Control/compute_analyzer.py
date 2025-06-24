from pathlib import Path
import h5py
import spikeinterface.full as si 

def compute_analyzer(rec_path, save_root, well_id, num_units_to_use='all', hp_cutoff_freq=300):
    """
    Computes the analyzer for a given recording and sorting.
    
    Parameters:
    - rec_path: Path to the recording file.
    - save_root: Root directory where the sorting results are saved.
    - well_id: Identifier for the well to be processed.
    - num_units_to_use: Number of units to restrict the sorting to.
    
    Returns:
    - analyzer: The computed analyzer object.
    """
    
    # Load the sorting
    path_to_sorting = Path(save_root) / well_id / 'sorter_output'
    sorting_train = si.read_kilosort(folder_path=path_to_sorting)

    # Load the recording
    h5 = h5py.File(rec_path)
    rec_name = list(h5['wells'][well_id].keys())[0]
    rec_train = si.MaxwellRecordingExtractor(rec_path, stream_id=well_id, rec_name=rec_name)

    # High-pass filter the recording
    hp_cutoff_frequency = hp_cutoff_freq  # Hz
    print(f"Applying high-pass filter with {hp_cutoff_frequency} Hz cut-off...")
    rec_train = si.highpass_filter(rec_train, freq_min=hp_cutoff_frequency)

    # Restrict sorting to a certain number of units
    if num_units_to_use != 'all':
        all_unit_ids = sorting_train.get_unit_ids()
        if num_units_to_use > len(all_unit_ids):
            num_units_to_use = len(all_unit_ids)
        selected_unit_ids = all_unit_ids[:num_units_to_use]  # Select the first 'num_units_to_use'
        sorting_train = sorting_train.select_units(unit_ids=selected_unit_ids)
    
        print(f"Restricted sorting to {len(sorting_train.get_unit_ids())} units for development.")

    # Create and compute analyzer with the scaled recording and restricted sorting
    analyzer = si.create_sorting_analyzer(sorting=sorting_train, recording=rec_train)
    analyzer.compute(['noise_levels', 'random_spikes', 'waveforms', 'templates'])
    analyzer.compute(['spike_locations', 'spike_amplitudes', 'correlograms', 
                      'principal_components', 'quality_metrics', 'template_metrics'])
    
    return rec_train, sorting_train, analyzer
