#!/usr/bin/env python3
"""
Final version of Eurostat time series extractor with flat indexes
"""

import os
import json
import gzip
import pandas as pd
from pathlib import Path
import sys
import time
import itertools

# Base paths
data_dir = Path("C:/Users/Antonio/sdmx/eurostat_sdmx/data")
cache_dir = data_dir / "cache"
datasets_dir = data_dir / "datasets"
exports_dir = data_dir / "exports"

# Create exports directory if it doesn't exist
exports_dir.mkdir(exist_ok=True, parents=True)

def extract_time_series(dataset_id, filters=None, debug=True):
    """Extract time series from a dataset with flat indexes"""
    if debug:
        print(f"Extracting time series for {dataset_id}")
    
    # Load metadata
    metadata_file = datasets_dir / f"{dataset_id}_metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load cached data
    data_file = cache_dir / f"{dataset_id}_data.json.gz"
    try:
        with gzip.open(data_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            if debug:
                print("Successfully loaded data with gzip.open")
    except Exception as e:
        if debug:
            print(f"Error loading with gzip.open: {e}")
        try:
            with open(data_file, 'rb') as f:
                compressed = f.read()
            decompressed = gzip.decompress(compressed)
            text = decompressed.decode('utf-8')
            data = json.loads(text)
            if debug:
                print("Successfully loaded data with manual decompression")
        except Exception as e:
            if debug:
                print(f"Error with manual decompression: {e}")
                print("Trying to read as plain file...")
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if debug:
                        print("Successfully loaded as plain JSON")
            except Exception as e:
                if debug:
                    print(f"Failed to load data: {e}")
                return None
    
    # If no filters provided, use default
    if filters is None:
        filters = {'freq': 'A', 'itm_newa': '40000', 'geo': 'IT'}
    
    if debug:
        print(f"Using filters: {filters}")
    
    # Extract dimensions from metadata
    meta_dimensions = {}
    for dim in metadata.get('dimensions', []):
        dim_id = dim.get('id')
        dim_pos = dim.get('position')
        dim_values = dim.get('values', {})
        meta_dimensions[dim_id] = {
            'position': dim_pos,
            'values': dim_values
        }
    
    # Get dimension order and size from data
    dimension_order = data.get('id', [])
    dimension_sizes = data.get('size', [])
    
    if not dimension_order or not dimension_sizes or len(dimension_order) != len(dimension_sizes):
        if debug:
            print("Error: Invalid dimension order or size information")
        return None
    
    if debug:
        print(f"Dimension order: {dimension_order}")
        print(f"Dimension sizes: {dimension_sizes}")
    
    # Find time dimension (might be named differently in data vs metadata)
    time_dim_meta = None
    time_dim_data = None
    
    # Look for time dimension in metadata
    for dim_id in meta_dimensions.keys():
        if dim_id.lower() in ('time_period', 'time'):
            time_dim_meta = dim_id
            break
    
    # Look for time dimension in data
    for dim_id in dimension_order:
        if dim_id.lower() in ('time_period', 'time'):
            time_dim_data = dim_id
            break
    
    if not time_dim_meta or not time_dim_data:
        if debug:
            print(f"Error: Time dimension not found. Metadata: {time_dim_meta}, Data: {time_dim_data}")
        return None
    
    if debug:
        print(f"Time dimension in metadata: {time_dim_meta}")
        print(f"Time dimension in data: {time_dim_data}")
    
    # Build a lookup to convert dimension values to indices
    dim_indices = {}
    for dim_id in dimension_order:
        if dim_id not in data.get('dimension', {}):
            if debug:
                print(f"Warning: Dimension {dim_id} not found in data")
            continue
            
        dim_data = data['dimension'][dim_id]
        if 'category' not in dim_data or 'index' not in dim_data['category']:
            if debug:
                print(f"Warning: No category index for dimension {dim_id}")
            continue
            
        index_map = dim_data['category']['index']
        if isinstance(index_map, dict):
            dim_indices[dim_id] = {code: idx for code, idx in index_map.items()}
        else:
            if debug:
                print(f"Warning: Unexpected index format for dimension {dim_id}")
    
    # For each filter, get the corresponding index
    filter_indices = {}
    for meta_dim_id, filter_value in filters.items():
        # Find matching data dimension
        data_dim_id = None
        for dim_id in dimension_order:
            if dim_id.lower() == meta_dim_id.lower():
                data_dim_id = dim_id
                break
        
        if not data_dim_id:
            if debug:
                print(f"Warning: Dimension {meta_dim_id} from metadata not found in data")
            continue
            
        if data_dim_id not in dim_indices:
            if debug:
                print(f"Warning: No index information for dimension {data_dim_id}")
            continue
            
        indices = dim_indices[data_dim_id]
        if filter_value not in indices:
            if debug:
                print(f"Warning: Value {filter_value} not found in dimension {data_dim_id}")
            continue
            
        filter_indices[data_dim_id] = indices[filter_value]
    
    if debug:
        print(f"Filter indices: {filter_indices}")
    
    # Collect all values for time dimension
    time_values = {}
    if time_dim_data in dim_indices:
        for code, idx in dim_indices[time_dim_data].items():
            # Try to get label from the data
            label = None
            time_dim_category = data['dimension'][time_dim_data].get('category', {})
            if 'label' in time_dim_category and code in time_dim_category['label']:
                label = time_dim_category['label'][code]
            
            # If not found, try to get from metadata
            if label is None and time_dim_meta in meta_dimensions:
                meta_values = meta_dimensions[time_dim_meta]['values']
                if code in meta_values:
                    label = meta_values[code]
            
            # Fallback to code
            if label is None:
                label = code
                
            time_values[idx] = label
    
    if debug:
        print(f"Found {len(time_values)} time periods")
        if len(time_values) > 0:
            print(f"First few: {list(time_values.items())[:3]}")
    
    # Now we need to calculate flat indices for each time value while keeping other dimensions fixed
    results = []
    
    # Calculate the "step sizes" for each dimension
    # In a flat index, each dimension has a multiplier based on the sizes of the following dimensions
    multipliers = [1] * len(dimension_order)
    for i in range(len(dimension_order) - 2, -1, -1):
        multipliers[i] = multipliers[i+1] * dimension_sizes[i+1]
    
    if debug:
        print(f"Dimension multipliers: {multipliers}")
    
    # For each time period
    for time_idx, time_label in time_values.items():
        # Create a position array with our filter values
        position = [0] * len(dimension_order)
        
        # Fill in filter values
        for dim_id, idx in filter_indices.items():
            if dim_id in dimension_order:
                dim_pos = dimension_order.index(dim_id)
                position[dim_pos] = idx
        
        # Fill in time value
        time_pos = dimension_order.index(time_dim_data)
        position[time_pos] = time_idx
        
        # Calculate flat index directly
        flat_idx = 0
        for i in range(len(position)):
            flat_idx += position[i] * multipliers[i]
        
        # Check if this index exists in the data
        str_idx = str(flat_idx)
        if str_idx in data['value']:
            value = data['value'][str_idx]
            results.append({
                'period': time_label,
                'value': value
            })
        elif debug and len(results) == 0:
            # For debugging: check a range around the calculated index
            for i in range(flat_idx - 5, flat_idx + 6):
                if str(i) in data['value']:
                    print(f"  Found nearby index: {i} (offset {i - flat_idx}) -> {data['value'][str(i)]}")
    
    # Create and save time series
    if results:
        df = pd.DataFrame(results)
        df.set_index('period', inplace=True)
        
        # Try to convert index to numeric and sort
        try:
            df.index = pd.to_numeric(df.index)
            df = df.sort_index()
        except:
            # If we can't convert to numeric, try to sort as strings
            try:
                df = df.sort_index()
            except:
                if debug:
                    print("Warning: Could not sort index")
        
        if debug:
            print(f"\nExtracted {len(results)} data points:")
            print(df)
        
        # Export to CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filter_str = "_".join([f"{k}_{v}" for k, v in filters.items()])
        export_path = exports_dir / f"{dataset_id}_{filter_str}_{timestamp}.csv"
        df.to_csv(export_path)
        if debug:
            print(f"\nExported to {export_path}")
        
        return df
    else:
        if debug:
            print("\nNo data found for the specified filters")
            
            # For debugging - try to dump a few raw values
            print("\nSample of raw values:")
            sample_count = 0
            for idx, value in list(data['value'].items())[:5]:
                print(f"  Index {idx}: {value}")
                sample_count += 1
                
        return None

def extract_all_time_series(dataset_id):
    """Extract all possible time series from a dataset"""
    print(f"Extracting all time series for {dataset_id}")
    
    # Load metadata
    metadata_file = datasets_dir / f"{dataset_id}_metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Extract all dimension combinations except TIME_PERIOD
    dimensions = {}
    time_dim = None
    for dim in metadata.get('dimensions', []):
        dim_id = dim.get('id')
        if dim_id.lower() in ('time_period', 'time'):
            time_dim = dim_id
            continue
            
        dim_values = dim.get('values', {})
        dimensions[dim_id] = list(dim_values.keys())
    
    if not time_dim:
        print("Error: TIME_PERIOD dimension not found")
        return
    
    # Generate all combinations
    combinations = []
    
    # Get all dimension IDs and their values
    dim_ids = list(dimensions.keys())
    dim_values = [dimensions[dim_id] for dim_id in dim_ids]
    
    # Generate product of all values
    for values in itertools.product(*dim_values):
        combo = {dim_ids[i]: values[i] for i in range(len(dim_ids))}
        combinations.append(combo)
    
    print(f"Generated {len(combinations)} filter combinations")
    
    # Extract time series for each combination
    successful = 0
    for i, filters in enumerate(combinations):
        print(f"\nProcessing combination {i+1}/{len(combinations)}: {filters}")
        # Turn off detailed debugging for all except first few
        debug_mode = (i < 3 or successful < 2)
        ts = extract_time_series(dataset_id, filters, debug=debug_mode)
        if ts is not None:
            successful += 1
    
    print(f"\nExtracted {successful} time series out of {len(combinations)} combinations")

# Run the extraction for a specific combination for detailed debugging
#extract_time_series('AACT_ALI01', {'freq': 'A', 'itm_newa': '40000', 'geo': 'IT'})

# Run extraction for all combinations
extract_all_time_series('AACT_ALI01')