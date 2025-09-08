from collections import defaultdict
import numpy as np
import glob
import os
import pickle
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress harmless runtime warnings

def compute_betti_numbers(graph_matrix, percentage=95):
    """
    Compute Betti numbers from adjacency matrix
    
    Parameters:
        graph_matrix: Adjacency matrix
        threshold: Threshold for converting matrix to graph
        
    Returns:
        dict: Dictionary containing Betti0 and Betti1
    """
    try:
        # 1. Apply threshold to create graph structure
        graph = defaultdict(list)
        n = graph_matrix.shape[0]
        threshold = np.percentile(graph_matrix, percentage)
        # target_density = 0.1
        # sorted_vals = np.sort(graph_matrix.flatten())
        # threshold = sorted_vals[int((1 - target_density) * len(sorted_vals))]

        for i in range(n):
            for j in range(i+1, n):  # Avoid duplicates and undirected graph bidirectional edges
                if graph_matrix[i,j] >= threshold:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # 2. Calculate number of connected components (Betti0)
        visited = set()
        components = 0
        
        for node in graph:
            if node not in visited:
                components += 1
                # Use BFS to traverse connected component
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        queue.extend(graph[current])
        
        # 3. Calculate Betti1 (number of cycles)
        # Using Euler's formula: Betti1 = edges - vertices + components
        num_vertices = len(graph)
        num_edges = sum(len(edges) for edges in graph.values()) // 2  # Undirected graph counts each edge twice
        
        betti_1 = num_edges - num_vertices + components
        
        return {
            'betti_0': components,
            'betti_1': max(betti_1, 0)  # Ensure non-negative
        }
        
    except Exception as e:
        print(f"Error computing Betti numbers: {str(e)}")
        return {
            'betti_0': np.nan,
            'betti_1': np.nan
        }

def topological_analysis(network, percentage=95):
    """Graph theory based topological analysis function"""
    try:
        # 1. Data preprocessing
        network = np.abs(network)
        network = (network - np.nanmin(network)) / (np.nanmax(network) - np.nanmin(network) + 1e-6)
        
        # 2. Compute Betti numbers
        betti_result = compute_betti_numbers(network, percentage)
        
        return {
            'betti_0': betti_result['betti_0'],
            'betti_1': betti_result['betti_1']
        }
    except Exception as e:
        print(f"Topological analysis error: {str(e)}")
        return {
            'betti_0': np.nan,
            'betti_1': np.nan
        }

def analyze_topological_features(data, percentage=95):
    """Final analysis pipeline with complete edge case handling"""
    all_results = {
        'rest': defaultdict(dict),
        'conditions': defaultdict(dict),
        'comparisons': defaultdict(dict)
    }
    
    # 1. Process Rest condition
    for source, matrices in data['rest'].items():
        try:
            avg_matrix = np.nanmean(matrices, axis=0)
            all_results['rest'][source] = topological_analysis(avg_matrix, percentage)
        except:
            all_results['rest'][source] = {
                'betti_0': np.nan,
                'betti_1': np.nan
            }
    
    # 2. Process each condition
    for cond, source_dict in data['conditions'].items():
        for source, matrices in source_dict.items():
            try:
                avg_matrix = np.nanmean(matrices, axis=0)
                all_results['conditions'][cond][source] = topological_analysis(avg_matrix, percentage)
            except:
                all_results['conditions'][cond][source] = {
                    'betti_0': np.nan,
                    'betti_1': np.nan
                }
    
    # 3. Safer comparison calculation (only Betti numbers)
    for cond in data['conditions']:
        common_sources = set(all_results['rest'].keys()) & set(all_results['conditions'][cond].keys())
        
        betti_0_diffs = []
        betti_1_diffs = []
        
        for src in common_sources:
            rest_res = all_results['rest'][src]
            cond_res = all_results['conditions'][cond][src]
            
            # Ensure all values are valid
            if all(not np.isnan(rest_res[k]) and not np.isnan(cond_res[k]) for k in ['betti_0', 'betti_1']):
                betti_0_diffs.append(cond_res['betti_0'] - rest_res['betti_0'])
                betti_1_diffs.append(cond_res['betti_1'] - rest_res['betti_1'])
        
        # Ensure lists are not empty
        betti_0_diffs = betti_0_diffs if betti_0_diffs else [np.nan]
        betti_1_diffs = betti_1_diffs if betti_1_diffs else [np.nan]
        
        all_results['comparisons'][cond] = {
            'betti_0_diff': betti_0_diffs,
            'betti_1_diff': betti_1_diffs,
            'mean_betti_0_diff': np.nanmean(betti_0_diffs),
            'mean_betti_1_diff': np.nanmean(betti_1_diffs),
            'n_sources': len(common_sources)
        }
    
    return all_results

def load_network_data(results_folder='results', network_type='TE', atom_type='xty'):
    """
    Load network data and organize into structured dictionary
    
    Parameters:
        results_folder: Path containing .pkl result files
        network_type: Network type ('TE' or 'PhiID')
        atom_type: For PhiID networks, specify atom type
        
    Returns:
        dict: {
            'rest': {source: [matrix1, matrix2,...]},  # Rest condition data
            'conditions': {cond_name: {source: [matrix1, matrix2,...]}},  # Other condition data
            'roi_names': list,  # List of ROI names
            'metadata': {
                'network_type': str,
                'atom_type': str
            }
        }
    """
    print("\n" + "="*50)
    print("Starting network data loading")
    print(f"Results folder: {results_folder}")
    print(f"Network type: {network_type}")
    print(f"Atom type: {atom_type}")
    print("="*50)
    
    result_files = glob.glob(os.path.join(results_folder, '*.pkl'))
    if not result_files:
        raise FileNotFoundError(f"No result files (.pkl) found in folder {results_folder}")
    
    print(f"\nFound {len(result_files)} result files:")
    for i, f in enumerate(result_files[:5]):
        print(f"  {i+1}. {os.path.basename(f)}")
    if len(result_files) > 5:
        print(f"  ... (Total {len(result_files)} files)")
    
    data = {
        'rest': defaultdict(list),
        'conditions': defaultdict(lambda: defaultdict(list)),
        'roi_names': None,
        'metadata': {
            'network_type': network_type,
            'atom_type': atom_type
        }
    }
    
    for file_idx, filepath in enumerate(sorted(result_files), 1):
        print(f"\nProcessing file {file_idx}/{len(result_files)}: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            # Get condition name
            cond = results.get('metadata', {}).get('selected_conditions', os.path.basename(filepath))
            if isinstance(cond, list):
                cond = cond[0] if len(cond) > 0 else os.path.basename(filepath)
            print(f"Condition name: {cond}")
            
            # Get ROI information
            if data['roi_names'] is None:
                data['roi_names'] = results['metadata']['roi_names']
                n_rois = len(data['roi_names'])
                print(f"Number of ROIs: {n_rois}")
                print(f"First 5 ROI names: {data['roi_names'][:5]}")
            
            # Collect matrices for all sources in this condition
            source_matrices = defaultdict(list)
            roi_pairs = results.get('conditions', {}).get(cond, {}).get('roi_pairs', {})
            print(f"Number of ROI pairs: {len(roi_pairs)}")
            
            for pair_idx, (pair_key, pair_data) in enumerate(roi_pairs.items(), 1):
                if not isinstance(pair_data, dict) or 'error' in pair_data:
                    continue
                
                # Get trial_sources for current ROI pair
                sources = []
                if network_type == 'TE' and 'TE' in pair_data and 'trial_sources' in pair_data['TE']:
                    sources = pair_data['TE']['trial_sources']
                elif network_type == 'PhiID' and 'PhiID' in pair_data and 'trial_sources' in pair_data['PhiID']:
                    sources = pair_data['PhiID']['trial_sources']
                elif 'PID' in pair_data and 'trial_sources' in pair_data['PID']:
                    sources = pair_data['PID']['trial_sources']
                
                if not sources:
                    continue
                    
                # Get trial-level values
                if network_type == 'TE':
                    trial_values = pair_data.get('TE', {}).get('values', [])
                elif network_type == 'PhiID':
                    trial_values = pair_data.get('PhiID', {}).get('trial_values', {}).get(atom_type, [])
                else:
                    trial_values = []
                
                # Build matrix for each trial
                src_idx, tgt_idx = pair_data['roi_info']['indices']
                for trial_idx, val in enumerate(trial_values):
                    if trial_idx >= len(sources):
                        continue
                    source = sources[trial_idx]
                    # Ensure enough matrices exist
                    while len(source_matrices[source]) <= trial_idx:
                        source_matrices[source].append(np.zeros((n_rois, n_rois)))
                    source_matrices[source][trial_idx][src_idx, tgt_idx] = val
            
            # Separate Rest from other conditions
            if 'rest' in cond.lower():
                print(f"Identified as Rest condition")
                data['rest'] = source_matrices
            else:
                print(f"Identified as experimental condition: {cond}")
                data['conditions'][cond] = source_matrices
            
            # Print loading statistics for current file
            print(f"Loaded sources: {len(source_matrices)}")
            for src, matrices in list(source_matrices.items())[:3]:
                print(f"  source '{src}': {len(matrices)} matrices")
            if len(source_matrices) > 3:
                print(f"  ... (Total {len(source_matrices)} sources)")
                
        except Exception as e:
            print(f"Error processing file {os.path.basename(filepath)}: {str(e)}")
            continue
    
    # Validate data completeness
    print("\n>>> Data completeness check <<<")
    if not data['rest']:
        raise ValueError("No Rest condition data found")
    else:
        print(f"Rest condition data loaded successfully, contains {len(data['rest'])} sources")
    
    if not data['conditions']:
        raise ValueError("No other condition data found")
    else:
        print(f"Loaded {len(data['conditions'])} experimental conditions:")
        for cond, sources in data['conditions'].items():
            print(f"  {cond}: {len(sources)} sources")
    
    print("\nData loading complete!")
    return data