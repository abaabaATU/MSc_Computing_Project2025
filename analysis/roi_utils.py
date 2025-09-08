# roi_utils.py

def extract_prefix(name):
    """
    Smart prefix extraction rules:
    1. If '-' exists, extract from start to '-' (including '-')
    2. Otherwise extract leading consecutive uppercase letters
    """
    if '-' in name:
        return name.split('/')[0].split('(')[0].split('-')[0] + '-'
    else:
        # Extract leading uppercase letters
        prefix = []
        for char in name:
            if char.isupper():
                prefix.append(char)
            else:
                break
        return ''.join(prefix)

def find_stim_roi_index(stim_roi, roi_names):
    """
    Exact matching of stim_roi to indices in roi_names, with improved prefix extraction logic
    
    Args:
        stim_roi: ROI name from stimulation data
        roi_names: List of all available ROI names
        
    Returns:
        List of matching indices or None if no match found
    """
    # Case 1: Direct match
    if stim_roi in roi_names:
        return [roi_names.index(stim_roi)]
    
    # Preprocessing
    has_bilateral = '(Bilateral)' in stim_roi
    bracket_content = ''
    base_name = stim_roi.split('(')[0].strip() if '(' in stim_roi else stim_roi
    
    if '(' in stim_roi and ')' in stim_roi:
        bracket_content = stim_roi.split('(')[1].split(')')[0].strip()
    
    # Extract prefix (new logic)
    prefix = extract_prefix(base_name)
    # Remove prefix part (keep content after prefix)
    base_name_without_prefix = base_name[len(prefix):]

    # Case 3: Contains both '/' and Bilateral
    if '/' in base_name_without_prefix and has_bilateral:
        # Split at slash (ensuring single split)
        left_part, right_part = base_name_without_prefix.split('/', 1)
        
        # Keep original characters (no length restriction)
        suffixes = [left_part.strip(), right_part.split()[0].strip()]
        
        matches = []
        for suffix_char in suffixes:
            for side in ['L', 'R']:
                candidate = f"{prefix}{suffix_char} ({side})"
                if candidate in roi_names:
                    matches.append(roi_names.index(candidate))
        if matches:
            return matches
    
    # Case 2: Contains only '/'
    elif '/' in base_name:
        # Split at slash (ensuring single split)
        left_part, right_part = base_name_without_prefix.split('/', 1)
        
        # Keep original characters (no length restriction)
        suffixes = [left_part.strip(), right_part.split()[0].strip()]
        
        side = bracket_content if bracket_content in ['L', 'R'] else 'L'
        
        matches = []
        for suffix_char in suffixes:
            candidate = f"{prefix}{suffix_char} ({side})"
            if candidate in roi_names:
                matches.append(roi_names.index(candidate))
        if matches:
            return matches
    
    # Case 4: Uppercase prefix matching
    elif base_name.split(' ')[0].isupper() and '(' in stim_roi:
        side = bracket_content if bracket_content in ['L', 'R'] else 'L'
        
        matches = [
            i for i, name in enumerate(roi_names)
            if name.startswith(prefix) and name.endswith(f"({side})")
        ]
        if matches:
            return matches
    
    # No match found
    return None

# brain_regions.py

# Default color mapping for groups
GROUP_COLORS = {
    # Sensory system (blue shades)
    'Visual': '#1f77b4',       # Primary visual - dark blue
    'Auditory': '#aec7e8',     # Auditory - light blue
    'Somatosensory': '#4b78c2', # Somatosensory - medium blue
    'Gustatory': '#7aa6db',    # Gustatory - sky blue
    'Visceral': '#5c8ac6',     # Visceral sensation - cobalt blue
    
    # Motor system (red shades)
    'Motor': '#d62728',        # Motor cortex - true red
    'Basal Ganglia': '#ff6b6b', # Basal ganglia - bright red
    
    # Association cortex (orange shades)
    'Prefrontal': '#ff7f0e',   # Prefrontal - standard orange
    'Orbital': '#ff9e4a',      # Orbitofrontal - light orange
    'Insular': '#ff8c00',      # Insular - dark orange
    'Temporal Assoc': '#ffaa5e', # Temporal association - honey orange
    
    # Subcortical structures (green shades)
    'Thalamus': '#2ca02c',     # Thalamus - dark green
    'Midbrain': '#59a869',     # Midbrain - grass green
    'Epithalamus': '#3d8c40',  # Epithalamus - olive green
    'Geniculate': '#7fbf7b',   # Geniculate nuclei - light green
    'Reticular': '#4da64d',    # Reticular formation - apple green
    
    # Other
    'Retrosplenial': '#9467bd', # Retrosplenial cortex - purple
    'Other': '#7f7f7f'        # Other - gray
}

# Functional group definitions
FUNCTIONAL_GROUPS = {
    # Sensory systems
    'Visual': ['VIS', 'VISa', 'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl', 'VISpm', 'VISpor', 'VISrl'],
    'Auditory': ['AUD', 'AUDd', 'AUDp', 'AUDpo', 'AUDv'],
    'Somatosensory': ['SS', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul', 'SSp-un', 'SSs'],
    'Gustatory': ['GU'],
    'Visceral': ['VISC'],
    
    # Motor systems
    'Motor': ['MO', 'MOp', 'MOs'],
    'Basal Ganglia': ['STR', 'PAL'],
    
    # Association cortices
    'Prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA'],
    'Orbital': ['ORB', 'ORBl', 'ORBm', 'ORBvl'],
    'Insular': ['AI', 'AId', 'AIv', 'AIp'],
    'Temporal Assoc': ['TEa', 'PERI', 'ECT'],
    
    # Subcortical structures
    'Thalamus': ['ATN', 'MTN', 'VENT', 'LAT', 'MED', 'ILM', 'RT'],
    'Midbrain': ['SCs', 'SCm', 'PAG', 'SNr', 'VTA', 'MRN', 'RN', 'PPN'],
    'Epithalamus': ['EPI'],
    'Geniculate': ['GENd', 'GENv'],
    'Reticular': ['PRT', 'CUN'],
    
    # Other
    'Retrosplenial': ['RSP', 'RSPagl', 'RSPd', 'RSPv'],
    'Other': []  # Kept empty to ensure all regions are classified
}

def auto_detect_groups(names):
    """Automatically classify brain regions based on their functional groups.
    
    Args:
        names: List of brain region names (may contain side markers like (L)/(R))
    
    Returns:
        Dictionary mapping indices to group names
    """
    groups = {}
    for i, name in enumerate(names):
        # Sensory systems
        if any(kw in name for kw in FUNCTIONAL_GROUPS['Visual']):
            groups[i] = 'Visual'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Auditory']):
            groups[i] = 'Auditory'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Somatosensory']):
            groups[i] = 'Somatosensory'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Gustatory']):
            groups[i] = 'Gustatory'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Visceral']):
            groups[i] = 'Visceral'
            
        # Motor systems
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Motor']):
            groups[i] = 'Motor'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Basal Ganglia']):
            groups[i] = 'Basal Ganglia'
            
        # Association cortices
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Prefrontal']):
            groups[i] = 'Prefrontal'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Orbital']):
            groups[i] = 'Orbital'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Insular']):
            groups[i] = 'Insular'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Temporal Assoc']):
            groups[i] = 'Temporal Assoc'
            
        # Subcortical structures
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Thalamus']):
            groups[i] = 'Thalamus'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Midbrain']):
            groups[i] = 'Midbrain'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Epithalamus']):
            groups[i] = 'Epithalamus'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Geniculate']):
            groups[i] = 'Geniculate'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Reticular']):
            groups[i] = 'Reticular'
            
        # Other
        elif any(kw in name for kw in FUNCTIONAL_GROUPS['Retrosplenial']):
            groups[i] = 'Retrosplenial'
        else:
            groups[i] = 'Other'

    # 检查是否有单节点社区
    from collections import Counter
    group_counts = Counter(groups.values())
    print("Group sizes:", group_counts)

    # 如果有单节点社区，合并到其他组
    if min(group_counts.values()) == 1:
        for node, group in groups.items():
            if group_counts[group] == 1:
                groups[node] = 'Other'  # 合并到默认组

    return groups

GROUP_COLORS_CONSOLIDATED = {
    # Sensory-Motor System
    'Sensory': '#1f77b4',       # All sensory modalities (blue)
    'Motor': '#d62728',        # Motor systems (red)
    
    # Higher Cognition
    'Association': '#ff7f0e',  # All association cortices (orange)
    
    # Subcortical
    'Subcortical': '#2ca02c',  # All subcortical structures (green)
    
    # Special Systems
    'Limbic': '#9467bd',       # Limbic system (purple)
    'Other': '#7f7f7f'         # Unclassified (gray)
}

FUNCTIONAL_GROUPS_CONSOLIDATED = {
    # Sensory systems (all modalities combined)
    'Sensory': [
        # Visual
        'VIS', 'VISa', 'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl', 'VISpm', 'VISpor', 'VISrl',
        # Auditory 
        'AUD', 'AUDd', 'AUDp', 'AUDpo', 'AUDv',
        # Somatosensory
        'SS', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul', 'SSp-un', 'SSs',
        # Other senses
        'GU', 'VISC'
    ],
    
    # Motor systems (including basal ganglia)
    'Motor': ['MO', 'MOp', 'MOs', 'STR', 'PAL'],
    
    # Association cortices (all combined)
    'Association': [
        'FRP', 'ACAd', 'ACAv', 'PL', 'ILA',  # Prefrontal
        'ORB', 'ORBl', 'ORBm', 'ORBvl',      # Orbital
        'AI', 'AId', 'AIv', 'AIp',           # Insular
        'TEa', 'PERI', 'ECT'                 # Temporal
    ],
    
    # Subcortical structures (all combined)
    'Subcortical': [
        'ATN', 'MTN', 'VENT', 'LAT', 'MED', 'ILM', 'RT',  # Thalamus
        'GENd', 'GENv',                                   # Geniculate
        'SCs', 'SCm', 'PAG', 'SNr', 'VTA', 'MRN', 'RN', 'PPN',  # Midbrain
        'PRT', 'CUN'                                      # Brainstem
    ],
    
    # Limbic system
    'Limbic': ['RSP', 'RSPagl', 'RSPd', 'RSPv', 'EPI'],
    
    # Other
    'Other': []
}

def auto_detect_consolidated(names, min_group_size=3):
    """Highly consolidated grouping with automatic small-group merging
    
    Args:
        names: List of region names
        min_group_size: Minimum number of regions required to maintain a separate group
        
    Returns:
        Dictionary mapping indices to consolidated group names
    """
    groups = {}
    for i, name in enumerate(names):
        if any(kw in name for kw in FUNCTIONAL_GROUPS_CONSOLIDATED['Sensory']):
            groups[i] = 'Sensory'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS_CONSOLIDATED['Motor']):
            groups[i] = 'Motor'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS_CONSOLIDATED['Association']):
            groups[i] = 'Association'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS_CONSOLIDATED['Subcortical']):
            groups[i] = 'Subcortical'
        elif any(kw in name for kw in FUNCTIONAL_GROUPS_CONSOLIDATED['Limbic']):
            groups[i] = 'Limbic'
        else:
            groups[i] = 'Other'
    
    # Auto-merge small groups
    from collections import Counter
    group_counts = Counter(groups.values())
    
    # Merge groups with fewer than min_group_size members into 'Other'
    if min_group_size > 0:
        groups = {
            k: v if group_counts[v] >= min_group_size else 'Other' 
            for k, v in groups.items()
        }
    
    print(f"Consolidated group counts: {Counter(groups.values())}")
    return groups

def normalize_roi_name(roi_name):
    """Remove quotes, (L)/(R) suffixes, and extra whitespace from ROI names"""
    name = str(roi_name).replace("'", "").replace('"', '').replace(' ','').strip()
    return name.replace('(L)', '').replace('(R)', '').replace('(Bilateral)','').strip()

# Mapping from experimental conditions to their corresponding ROI names
condition_to_rois = {
    'MOp (L)': ['MOp (L)'],
    'VISam/pm (R)': ['VISam (R)', 'VISpm (R)'],
    'AUD (L)': ['AUDd (L)', 'AUDp (L)', 'AUDpo (L)', 'AUDv (L)'],
    'SSp-ul/ll (R)': ['SSp-ul (R)', 'SSp-ll (R)'],
    'RSPd/v (Bilateral)': ['RSPd (L)', 'RSPd (R)', 'RSPv (L)', 'RSPv (R)'],
    'VISp (L)': ['VISp (L)'],
    'MOs (R)': ['MOs (R)'],
    'SSp-bfd (L)': ['SSp-bfd (L)'],
    'VISa/rl (R)': ['VISa (R)', 'VISrl (R)']
}

def find_stim_roi_index(condition, roi_names):
    # Get the list of ROI names corresponding to the given condition
    rois_for_condition = condition_to_rois.get(condition, [])
    # Return the indices of these ROIs in the roi_names list
    return [roi_names.index(roi) for roi in rois_for_condition if roi in roi_names]