import spikeinterface.extractors as se
import spikeinterface.full as si
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm # For the progress bar


def screen_maxtwo_activity(
    rec_path,
    segment_duration_s=2,
    rate_lower_threshold=0.01, # Minimum individual channel firing rate (Hz/channel) to display rate_lower_threshold
    rate_upper_threshold=20,   # Maximum individual channel firing rate (Hz/channel) to display
    amp_lower_threshold=5,     # Minimum individual absolute amplitude (uV) to display
    amp_upper_threshold=200    # Max...
):
    """
    Analyzes MaxTwo well plate data to provide a fast, rough overview of
    amplitudes and estimated firing rates as distributions (box plots) for each well.
    Individual data points (channel rates or spike amplitudes) below the specified
    thresholds will be filtered out before plotting.

    Parameters
    ----------
    rec_path : str
        Path to the data.raw.h5 file.
    segment_duration_s : int or float
        Duration of the segment (in seconds) to analyze from the beginning
        of each recording. Default is 2 seconds.
    rate_lower_threshold : float
        Individual channel firing rates below this value will not be included
        in the firing rate box plot. Set to 0 to show all.
    rate_upper_threshold : floar
        ...
    amp_lower_threshold : float
        Individual absolute amplitudes below this value will not be included
        in the amplitude box plot. Set to 0 to show all.
    amp_upper_threshold : float
        ...
    """

    try:
        with h5py.File(rec_path, 'r') as h5:
            well_ids = sorted(list(h5['wells'].keys()))
    except Exception as e:
        print(f"Error reading well IDs from {rec_path}: {e}")
        return pd.DataFrame(), {}, {}

    all_well_rate_distributions = {}
    all_well_amplitude_distributions = {}

    print(f"Found {len(well_ids)} wells in {rec_path}")
    print(f"Analyzing {segment_duration_s} seconds from each well for a rough overview.")

    for stream_id in tqdm(well_ids, desc="Processing Wells", colour="green"):
        try:
            well_rec_name = None
            with h5py.File(rec_path, 'r') as h5_inner:
                if stream_id in h5_inner['wells']:
                    well_rec_name = list(h5_inner['wells'][stream_id].keys())[0]
                else:
                    raise ValueError(f"Well ID '{stream_id}' not found in the H5 file structure.")

            if well_rec_name is None:
                raise ValueError(f"Could not determine recording name for well {stream_id}")

            rec = se.MaxwellRecordingExtractor(file_path=rec_path, stream_id=stream_id, rec_name=well_rec_name)

            sampling_frequency = rec.get_sampling_frequency()
            num_channels = rec.get_num_channels()
            num_total_samples = rec.get_num_samples(segment_index=0)

            segment_nsamples = int(segment_duration_s * sampling_frequency)
            if segment_nsamples > num_total_samples:
                segment_nsamples = num_total_samples
            
            recording_filtered = si.bandpass_filter(rec, freq_min=300, freq_max=4999) # freq_max < 5000
            traces = recording_filtered.get_traces(
                segment_index=0,
                start_frame=0,
                end_frame=segment_nsamples
            )

            if traces.size == 0 or traces.shape[0] == 0:
                tqdm.write(f"  Warning: No valid traces found for segment in well {stream_id}. Skipping.")
                all_well_rate_distributions[stream_id] = []
                all_well_amplitude_distributions[stream_id] = []
                continue

            noise_std = np.median(np.abs(traces)) / 0.6745
            threshold = 5 * noise_std # Threshold for detecting events

            current_well_amplitudes = []
            current_well_rates = []

            duration_s_actual = segment_nsamples / sampling_frequency

            for i in range(num_channels):
                channel_trace = traces[:, i]
                negative_crossings = np.where(channel_trace < -threshold)[0]
                channel_crossings = np.sort(negative_crossings)
                
                # Calculate per-channel rate and apply point threshold
                if duration_s_actual > 0:
                    per_channel_rate = len(channel_crossings) / duration_s_actual
                    if per_channel_rate >= rate_lower_threshold and per_channel_rate <= rate_upper_threshold: # Filter individual rate points
                        current_well_rates.append(per_channel_rate)

                # Collect all absolute peak amplitudes for this channel and apply point threshold
                if len(channel_crossings) > 0:
                    channel_amplitudes = np.abs(channel_trace[channel_crossings])
                    # Filter individual amplitude points
                    filtered_amplitudes = [amp for amp in channel_amplitudes if (amp >= amp_lower_threshold and amp <= amp_upper_threshold)]
                    current_well_amplitudes.extend(filtered_amplitudes)

            all_well_rate_distributions[stream_id] = current_well_rates
            all_well_amplitude_distributions[stream_id] = current_well_amplitudes

        except Exception as e:
            tqdm.write(f"  Error processing well {stream_id}: {e}. Skipping this well.")
            all_well_rate_distributions[stream_id] = []
            all_well_amplitude_distributions[stream_id] = []
            continue

    # Prepare data for plotting (no filtering of entire wells here, just ensure lists aren't empty for plotting)
    rate_data_for_boxplot = {}
    amplitude_data_for_boxplot = {}
    
    # Collect summary data for the DataFrame
    summary_data = []

    for well_id in well_ids:
        rates = all_well_rate_distributions.get(well_id, [])
        amps = all_well_amplitude_distributions.get(well_id, [])

        rate_data_for_boxplot[well_id] = rates
        amplitude_data_for_boxplot[well_id] = amps

        # Calculate summary statistics from the (potentially filtered) data
        summary_data.append({
            'well_id': well_id,
            'mean_firing_rate_Hz': np.mean(rates) if rates else np.nan,
            'median_firing_rate_Hz': np.median(rates) if rates else np.nan,
            'mean_amplitude_uV': np.mean(amps) if amps else np.nan,
            'median_amplitude_uV': np.median(amps) if amps else np.nan
        })
    
    df_summary = pd.DataFrame(summary_data)

    # Get the ordered well IDs for plotting, based on whether they have data points after filtering
    well_ids_for_rate_plot = [well_id for well_id in well_ids if well_id in rate_data_for_boxplot]
    rate_values_for_boxplot = [rate_data_for_boxplot[well_id] for well_id in well_ids_for_rate_plot]

    well_ids_for_amplitude_plot = [well_id for well_id in well_ids if well_id in amplitude_data_for_boxplot]
    amplitude_values_for_boxplot = [amplitude_data_for_boxplot[well_id] for well_id in well_ids_for_amplitude_plot]

    # --- Plotting the overview ---
    if well_ids_for_rate_plot or well_ids_for_amplitude_plot:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f'MaxTwo 24-Well Plate Overview: Firing Rate & Amplitude Distributions ({segment_duration_s}s segment)\n'
            f'(Individual Firing Rate between {rate_lower_threshold:.2f} Hz and {rate_upper_threshold:.2f} Hz ' 
            f'and Amplitude between {amp_lower_threshold:.1f} uV and {amp_upper_threshold:.1f} uV)', 
            fontsize=14
        )

        # Plot Firing Rates
        if rate_values_for_boxplot:
            axes[0].boxplot(rate_values_for_boxplot, labels=well_ids_for_rate_plot, showfliers=True)
            axes[0].set_title('Estimated Firing Rate Distribution per Well (Hz/channel)')
            axes[0].set_xlabel('Well ID')
            axes[0].set_ylabel('Estimated Firing Rate (Hz/channel)')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            # Add a horizontal line for the threshold for visual reference if desired
            axes[0].axhline(y=rate_lower_threshold, color='r', linestyle='--', alpha=0.6, label='Lower Threshold')
            axes[0].axhline(y=rate_upper_threshold, color='r', linestyle='--', alpha=0.6, label='Upper Threshold')
            axes[0].legend(loc='upper right')
        else:
            axes[0].text(0.5, 0.5, 'No firing rate data to plot above point threshold.', horizontalalignment='center', verticalalignment='center',
                         transform=axes[0].transAxes)
            axes[0].set_title('Estimated Firing Rate Distribution per Well')
            axes[0].set_xlabel('Well ID')
            axes[0].set_ylabel('Estimated Firing Rate (Hz/channel)')

        # Plot Amplitudes
        if amplitude_values_for_boxplot:
            axes[1].boxplot(amplitude_values_for_boxplot, labels=well_ids_for_amplitude_plot, showfliers=True)
            axes[1].set_title('Absolute Amplitude Distribution per Well (uV)')
            axes[1].set_xlabel('Well ID')
            axes[1].set_ylabel('Absolute Amplitude (uV)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
            axes[1].axhline(y=amp_lower_threshold, color='r', linestyle='--', alpha=0.6, label='Lower Threshold')
            axes[1].axhline(y=amp_upper_threshold, color='r', linestyle='--', alpha=0.6, label='Upper Threshold')
            axes[1].legend(loc='upper right')
        else:
            axes[1].text(0.5, 0.5, 'No amplitude data to plot above point threshold.', horizontalalignment='center', verticalalignment='center',
                         transform=axes[1].transAxes)
            axes[1].set_title('Absolute Amplitude Distribution per Well')
            axes[1].set_xlabel('Well ID')
            axes[1].set_ylabel('Absolute Amplitude (uV)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("No data to plot after applying point-wise thresholds.")

    return df_summary, all_well_rate_distributions, all_well_amplitude_distributions
