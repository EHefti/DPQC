from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import h5py
import spikeinterface.full as si 
import spikeinterface.curation as sc
import spikeinterface.widgets as sw
import spikeinterface.extractors as si_extractors
# import spikeinterface.core as si


def generate_cluster_info(path_to_sorting):
    cluster_info_path = path_to_sorting / 'cluster_info.tsv'
    if not cluster_info_path.exists():
        spike_clusters = np.load(path_to_sorting / 'spike_clusters.npy')
        unit_ids = np.unique(spike_clusters)
    
        # Create a DataFrame with just 'cluster_id'
        cluster_info_df = pd.DataFrame({'cluster_id': unit_ids})
    
        # Save to TSV
        cluster_info_df.to_csv(cluster_info_path, sep='\t', index=False)
        print(f"Generated basic cluster_info.tsv at: {cluster_info_path}")


def compute_analyzer(rec_path, save_root, well_id, num_units_to_use='all', hp_cutoff_freq=1):
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
    generate_cluster_info(path_to_sorting)
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

    all_extensions_to_compute = [
        'noise_levels', 'random_spikes', 'waveforms', 'templates',
        'spike_locations', 'spike_amplitudes', 'correlograms',
        'principal_components', 'quality_metrics', 'template_metrics'
    ]
    
    extension_specific_params = {
        'correlograms': {
            'bin_ms': 1.0,   # Use 'bin_ms' (milliseconds)
            'window_ms': 150.0 # Use 'window_ms' (milliseconds)
        }
    }
    
    analyzer.compute(
        all_extensions_to_compute,
        extension_params=extension_specific_params
    )
    
    return rec_train, sorting_train, analyzer



def interactive_unit_labeler(recording, sorting, analyzer, output_dir_for_labels='KiloSort_Quality_Control/'):
    """
    Launches an interactive GUI for manually labeling spike sorting units as 'good', 'MUA', 'noise', or 'unlabeled'.
    Labels can be exported to a TSV file.

    Parameters
    ----------
    recording : spikeinterface.extractors.BaseRecording
        The recording object.
    sorting : spikeinterface.sorters.BaseSorting
        The sorting object containing the units to be labeled.
    analyzer : spikeinterface.postprocessing.SortingAnalyzer
        The sorting analyzer object, pre-computed with waveforms and correlograms.
    output_dir_for_labels : str or Path, default='KiloSort_Quality_Control/'
        The directory where the 'manual_unit_labels.tsv' file will be saved.
        This directory will be created if it doesn't exist.
    """

    output_dir_for_labels = Path(output_dir_for_labels)
    output_dir_for_labels.mkdir(parents=True, exist_ok=True)

    # --- Initialize Unit Properties for Labeling ---
    unit_ids = sorting.get_unit_ids()

    # Ensure 'quality_label' property exists, initialize if not
    if 'quality_label' not in sorting.get_property_keys():
        initial_labels = ['unlabeled'] * len(unit_ids)
        sorting.set_property('quality_label', initial_labels)

    current_unit_labels = {unit_id: sorting.get_property('quality_label')[i] for i, unit_id in enumerate(unit_ids)}

    # --- Create Interactive Widgets ---
    unit_selector = widgets.Dropdown(
        options=unit_ids,
        value=unit_ids[0] if unit_ids.size > 0 else None,
        description='Select Unit:',
        disabled=False,
    )

    output_plot = widgets.Output()

    button_good = widgets.Button(description="Label as Good", button_style='success')
    button_nonsoma = widgets.Button(description="Label as Non-Somatic", button_style='warning')
    button_mua = widgets.Button(description="Label as MUA", button_style='warning')
    button_noise = widgets.Button(description="Label as Noise", button_style='danger')
    button_unlabeled = widgets.Button(description="Unlabel", button_style='info')
    button_export = widgets.Button(description="Export Labels to TSV", button_style='primary')

    label_status = widgets.Textarea(
        value='Labels will appear here.',
        description='Unit Labels:',
        disabled=True,
        layout=widgets.Layout(width='auto', height='150px')
    )

    # --- Define Update and Labeling Functions ---
    def update_plot(change):
        selected_unit_id = change['new']
        if selected_unit_id is None: # Handle case with no units
            with output_plot:
                clear_output(wait=True)
            return

        with output_plot:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Plot Waveforms
            spike_train = sorting.get_unit_spike_train(selected_unit_id)
            num_spikes = spike_train.size
            sw.plot_unit_templates(analyzer, unit_ids=[selected_unit_id], ax=axes[0], same_axis=True, scale=1)
            axes[0].set_title(f"Waveforms for Unit {selected_unit_id} (# Spikes: {num_spikes}, Label: {current_unit_labels[selected_unit_id]})")

            # Plot Autocorrelogram
            sw.plot_autocorrelograms(analyzer, unit_ids=[selected_unit_id], ax=axes[1])
            axes[1].set_title(f"Autocorrelogram for Unit {selected_unit_id}")
            
            plt.tight_layout()
            plt.show()

    def update_label_status():
        sorted_labels = sorted(current_unit_labels.items()) # Sort for consistent display
        label_text = "\n".join([f"Unit {uid}: {label}" for uid, label in sorted_labels])
        label_status.value = label_text

    def apply_label(b):
        selected_unit_id = unit_selector.value
        if selected_unit_id is None:
            print("No unit selected.")
            return

        label = b.description.split(" ")[2].lower()
        if label == "unlabel":
            label = "unlabeled"

        current_unit_labels[selected_unit_id] = label
        idx = np.where(unit_ids == selected_unit_id)[0][0]
        new_property_values = list(sorting.get_property('quality_label'))
        new_property_values[idx] = label
        sorting.set_property('quality_label', new_property_values)

        # Re-draw the plot for the current unit to reflect the new label
        update_plot({'new': selected_unit_id})
        update_label_status()
        print(f"Unit {selected_unit_id} labeled as '{label}'")

        # --- Go to next unit automatically ---
        current_unit_index = list(unit_ids).index(selected_unit_id)
        next_unit_index = current_unit_index + 1

        if next_unit_index < len(unit_ids):
            unit_selector.value = unit_ids[next_unit_index]
        else:
            print("All units reviewed!")
            # Optionally, you could reset to the first unit or disable the selector
            # unit_selector.value = unit_ids[0]
            # unit_selector.disabled = True

    def export_labels_to_tsv(b):
        """Exports the current unit labels to a TSV file."""
        labels_df = pd.DataFrame({
            'unit_id': list(current_unit_labels.keys()),
            'quality_label': list(current_unit_labels.values())
        })
        output_filepath = Path(output_dir_for_labels) / "manual_unit_labels.tsv"
        labels_df.to_csv(output_filepath, sep='\t', index=False)
        print(f"Labels exported to: {output_filepath}")

    # --- Connect Widgets to Functions ---
    if unit_selector.value is not None: # Only observe if there are units
        unit_selector.observe(update_plot, names='value')

    button_good.on_click(apply_label)
    button_nonsoma.on_click(apply_label)
    button_mua.on_click(apply_label)
    button_noise.on_click(apply_label)
    button_unlabeled.on_click(apply_label)
    button_export.on_click(export_labels_to_tsv)

    # Initial plot and label status update (only if there are units)
    if unit_ids.size > 0:
        update_plot({'new': unit_selector.value})
    update_label_status()

    # --- Arrange and Display Widgets ---
    label_buttons = widgets.HBox([button_good, button_nonsoma, button_mua, button_noise, button_unlabeled])
    export_button_box = widgets.HBox([button_export])
    ui = widgets.VBox([unit_selector, label_buttons, output_plot, label_status, export_button_box])

    print("Displaying interactive GUI...")
    display(ui)

    # Return the sorting object with updated properties in case it's needed externally
    return sorting



def generate_sample_data(durations=[10], sampling_frequency=30000, num_channels=4, seed=42):
    """
    Generates toy ground truth recording and sorting data, along with a pre-computed analyzer.

    Parameters
    ----------
    durations : list, default=[10]
        Durations of the recording segments in seconds.
    sampling_frequency : float, default=30000
        Sampling frequency of the recording.
    num_channels : int, default=4
        Number of channels in the recording.
    seed : int, default=42
        Seed for reproducibility of the toy data generation.

    Returns
    -------
    recording : spikeinterface.extractors.BaseRecording
        The generated toy recording object.
    sorting : spikeinterface.sorters.BaseSorting
        The generated toy ground truth sorting object.
    analyzer : spikeinterface.postprocessing.SortingAnalyzer
        The analyzer object with computed random_spikes, unit_waveforms, and correlograms.
    """
    print(f"Generating toy data with durations={durations}, sampling_frequency={sampling_frequency}, num_channels={num_channels}...")

    # Generate ground truth recording
    recording = si_extractors.toy_example.generate_ground_truth_recording(
        durations=durations, sampling_frequency=sampling_frequency, num_channels=num_channels, seed=seed
    )

    # Manually create spike times and clusters for a simple toy sorting
    # This example assumes a few distinct units with some spikes
    spike_times = np.array([
        1000, 1050, 1100,  # Unit 0
        2000, 2050, 2100,  # Unit 1
        3000, 3050,        # Unit 2
        4000, 4050         # Unit 3
    ])
    spike_clusters = np.array([
        0, 0, 0,
        1, 1, 1,
        2, 2,
        3, 3
    ])

    # Ensure spike_times are within the recording duration (simple check for toy data)
    max_time_point = int(durations[0] * sampling_frequency)
    valid_indices = spike_times < max_time_point
    spike_times = spike_times[valid_indices]
    spike_clusters = spike_clusters[valid_indices]


    sorting = NumpySorting.from_times_labels(spike_times, spike_clusters, sampling_frequency=recording.sampling_frequency)

    # Create a temporary folder for waveforms and analyzer computations
    temp_waveforms_folder = Path(tempfile.mkdtemp())
    print(f"Temporary folder for analyzer data: {temp_waveforms_folder}")

    # Create and compute analyzer properties
    analyzer = si.create_sorting_analyzer(sorting, recording, folder=temp_waveforms_folder, format="binary", sparse=False)
    analyzer.compute("random_spikes")
    analyzer.compute("unit_waveforms")
    analyzer.compute("correlograms")

    print(f"Generated {len(sorting.get_unit_ids())} units from toy data.")
    return recording, sorting, analyzer



def auto_label_stream(rec_path, save_root, stream_id, model_folder, hp_cutoff = 1):
    """
    Function to label a stream using the trained model. 
    Labels are saved in the sorting output directory.
    
    Parameters:
    - rec_path (str): Path to the recording file.
    - save_root (str): Root directory where the sorting output is saved.
    - stream_id (str): Identifier for the stream to be labeled.
    - model_folder (str): Folder where the trained model is stored.
    - hp_cutoff (float): Cutoff frequency for the recording dato to clean data
    
    Returns:
    - None
    """
    
    print("")
    print("------------------------")
    print("")
    print(f'Analyzing and Labeling {stream_id}:')

    """
    h5 = h5py.File(rec_path)
    rec_name = list(h5['wells'][stream_id].keys())[0]
    rec = si.MaxwellRecordingExtractor(rec_path, stream_id=stream_id, rec_name=rec_name)

    
    print(f"Applying high-pass filter with {hp_cutoff} Hz cut-off...")
    rec = si.highpass_filter(rec, freq_min=hp_cutoff)
    
    
    path_to_sorting = Path(save_root) / stream_id / 'sorter_output'
    generate_cluster_info(path_to_sorting)
    sorting = si.read_kilosort(folder_path=path_to_sorting)
    

    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec)
    analyzer.compute(['noise_levels','random_spikes','waveforms','templates','spike_locations','spike_amplitudes',
                      'correlograms','principal_components', 'quality_metrics', 'template_metrics'])
    
    """
    
    recording, sorting, analyzer = compute_analyzer(
        rec_path, save_root, 
        well_id=stream_id, 
        num_units_to_use='all', # use num_units_to_use='all' to use all units
        hp_cutoff_freq=hp_cutoff) 
    

    
    labels_and_probabilities = si.auto_label_units(
        sorting_analyzer=analyzer,
        model_folder=model_folder,
        trust_model=True
    )

    # print how many units were labeled as good and bad
    good_units = sum(label == 'good' for label in labels_and_probabilities["prediction"])
    bad_units = sum(label == 'bad' for label in labels_and_probabilities["prediction"])
    print(f'Number of good units: {good_units}')
    print(f'Number of bad units: {bad_units}')

    avg_confidence = mean(labels_and_probabilities["probability"])
    print(f'Average confidence of the model: {avg_confidence:.3f}')
    print("")

    labels_and_probabilities = pd.DataFrame(labels_and_probabilities)
    labels_path = Path(save_root) / stream_id / 'sorter_output' / 'labels_and_probabilities.tsv'
    labels_and_probabilities.to_csv(labels_path, sep='\t', index=False)



def plot_conf_matrix(manual_labels, predictions, label_classes, title='Predicted vs Manual Label', xlabel='Predicted Label', ylabel='Manual Label'):
    conf_matrix = confusion_matrix(manual_labels, predictions, labels=label_classes)
    balanced_accuracy = balanced_accuracy_score(manual_labels, predictions)
    plt.figure(figsize=(8, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, pad=20)
    plt.suptitle(f"Balanced Accuracy: {balanced_accuracy:.3f}", fontsize=18, weight='bold', y=0.98)
    plt.colorbar(label='Count', shrink=0.7)
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=14, weight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(ticks=np.arange(len(label_classes)), labels=label_classes, rotation=45, ha='right', fontsize=12)
    plt.yticks(ticks=np.arange(len(label_classes)), labels=label_classes, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()




def plot_model_evaluation(analyzer, model_folder, manual_labels):
    
    print("---Model Evaluation---")
    print("")
    model, model_info = sc.load_model(
        model_folder=model_folder,
        trusted=['numpy.dtype']
    )

    labels_and_probabilities = si.auto_label_units(
        sorting_analyzer=analyzer,
        model_folder=model_folder,
        trust_model=True
    )

    avg_confidence = mean(labels_and_probabilities["probability"])
    print(labels_and_probabilities.head())
    print('...')
    print(f'The average confidence of the model is {avg_confidence:.3f}.')
    print("")
    print("---------------")
    print("")

    predictions = labels_and_probabilities['prediction'].tolist()
    
    class_labels_ordered = sorted(list(model_info['label_conversion'].values()))

    if len(manual_labels) != len(predictions):
        print("Warning: Length of manual_labels and model predictions do not match. Cannot generate confusion matrix.")
        return

    plot_conf_matrix(manual_labels, predictions, label_classes=class_labels_ordered, 
                     title='Predicted vs Manual Label', xlabel='Predicted Label', ylabel='Manual Label')

    """
    conf_matrix = confusion_matrix(manual_labels, predictions, labels=class_labels_ordered)

    balanced_accuracy = balanced_accuracy_score(manual_labels, predictions)

    plt.figure(figsize=(7, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black",
                     fontsize=15, weight='bold')
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Manual Label', fontsize=12)
    
    plt.xticks(ticks=np.arange(len(class_labels_ordered)), labels=class_labels_ordered, rotation=45, ha='right', fontsize=10)
    plt.yticks(ticks=np.arange(len(class_labels_ordered)), labels=class_labels_ordered, fontsize=10)
    
    plt.title('Predicted vs Manual Label', fontsize=14, pad=20)
    plt.suptitle(f"Balanced Accuracy: {balanced_accuracy:.3f}", fontsize=16, weight='bold', y=0.98)
    plt.colorbar(label='Count', shrink=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    """


