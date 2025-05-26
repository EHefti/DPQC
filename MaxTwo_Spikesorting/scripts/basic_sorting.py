"""
Script Name: basic_sorting.py
Description: This script sorts MxW chips using Kilosort 2.5
Author: Philipp Hornauer
Date: 2024-12-03
"""

import sys
import argparse
import spikeinterface.full as si
import h5py

from tqdm import tqdm

sys.path.append("/home/phornauer/Git/axon_tracking/")
from axon_tracking import spike_sorting as ss

sorter = 'kilosort2_5'
si.Kilosort2_5Sorter.set_kilosort2_5_path('/home/phornauer/Git/Kilosort_2020b')
sorter_params = si.get_default_sorter_params(si.Kilosort2_5Sorter)
sorter_params['n_jobs'] = -1
sorter_params['detect_threshold'] = 6
sorter_params['minFR'] = 0.01
sorter_params['minfr_goodchannels'] = 0.01
sorter_params['keep_good_only'] = False
sorter_params['do_correction'] = False
sorter_params['NT'] = 64*1024 + 64

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process input data and save output.")
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--output", required=True, help="Path to save the sorter output.")
    args = parser.parse_args()
    
        # Run the main logic
    try:
        h5 = h5py.File(args.input)
        stream_ids = list(h5['wells'].keys())
        
        for stream_id in tqdm(stream_ids):
            try:
                rec_name = list(h5['wells'][stream_id].keys())[0]
                rec = si.MaxwellRecordingExtractor(args.input, stream_id=stream_id, rec_name=rec_name)
                ss.clean_sorting(rec, args.output, stream_id=stream_id, sorter=sorter, sorter_params=sorter_params)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
        print("Script executed successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
