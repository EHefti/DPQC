import os
import glob

def generate_path_list(root_path: str, 
                       path_logic: list = None, 
                       split_prefix: str = "") -> list[str]:
    if path_logic is None:
        path_logic = ['*', '*', 'w*']

    full_pattern = os.path.join(root_path, *path_logic)
    matching_paths = glob.glob(full_pattern)
    
    sorted_dirs = sorted(list(set(os.path.dirname(p) if os.path.isfile(p) else p for p in matching_paths)))
    
    sorting_path_list = []

    if not split_prefix:
        sorting_path_list = sorted_dirs
    else:
        for s_dir in sorted_dirs:
            dirs = glob.glob(os.path.join(s_dir, split_prefix))
            for d in dirs:
                if os.path.isdir(d):
                    sorting_path_list.append(d)
                elif os.path.isfile(d):
                    sorting_path_list.append(os.path.dirname(d))

    return sorting_path_list