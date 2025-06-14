import os

def infer_sampling_rate(sorting_path: str) -> float:
    params_path = os.path.join(sorting_path, "params.py")
    
    with open(params_path, 'r') as fileID:
        params_txt = fileID.read()
    
    params_parts = params_txt.split('\n')
    
    sampling_rate = None
    for part in params_parts:
        if part.strip().startswith('sample_rate'):
            sr_parts = part.split(' = ')
            if len(sr_parts) > 1:
                try:
                    sampling_rate = float(sr_parts[1].strip())
                    break
                except ValueError:
                    continue
    
    if sampling_rate is None:
        raise ValueError(f"Could not infer 'sample_rate' from {params_path}")
        
    return sampling_rate