#!/usr/bin/env python3
"""
Simple extractor to debug Eurostat data extraction
"""

import os
import json
import gzip
import pandas as pd
from pathlib import Path
import sys

# Base paths
data_dir = Path("C:/Users/Antonio/sdmx/eurostat_sdmx/data")
cache_dir = data_dir / "cache"
datasets_dir = data_dir / "datasets"
exports_dir = data_dir / "exports"

# Create exports directory if it doesn't exist
exports_dir.mkdir(exist_ok=True, parents=True)

def simple_extract(dataset_id):
    """Simple function to extract time series from a dataset"""
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
            print("Successfully loaded data with gzip.open")
    except Exception as e:
        print(f"Error loading with gzip.open: {e}")
        try:
            with open(data_file, 'rb') as f:
                compressed = f.read()
            decompressed = gzip.decompress(compressed)
            text = decompressed.decode('utf-8')
            data = json.loads(text)
            print("Successfully loaded data with manual decompression")
        except Exception as e:
            print(f"Error with manual decompression: {e}")
            print("Trying to read as plain file...")
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print("Successfully loaded as plain JSON")
            except Exception as e:
                print(f"Failed to load data: {e}")
                return
    
    # Print data structure
    print("\nData structure:")
    print(f"Keys: {list(data.keys())}")
    if 'id' in data:
        print(f"Dimension order: {data['id']}")
    if 'size' in data:
        print(f"Size: {data['size']}")
    if 'dimension' in data:
        print(f"Dimensions: {list(data['dimension'].keys())}")
    if 'value' in data:
        print(f"Number of values: {len(data['value'])}")
        print(f"Sample keys: {list(data['value'].keys())[:3]}")
    
    # Extract some sample values
    if 'value' in data and 'id' in data:
        # Get dimension order
        dimension_order = data['id']
        
        # Find TIME_PERIOD dimension position
        time_pos = -1
        for i, dim in enumerate(dimension_order):
            if dim == 'TIME_PERIOD' or dim.lower() == 'time':
                time_pos = i
                break
        
        if time_pos == -1:
            print("Error: TIME_PERIOD dimension not found")
            return
        
        print(f"\nTIME_PERIOD dimension found at position {time_pos}")
        
        # Extract dimensions from metadata
        dimensions = {}
        for dim in metadata.get('dimensions', []):
            dim_id = dim.get('id')
            dim_pos = dim.get('position')
            dim_values = dim.get('values', {})
            dimensions[dim_id] = {
                'position': dim_pos,
                'values': dim_values
            }
        
        # Try a specific combination that should work
        filters = {'freq': 'A', 'itm_newa': '40000', 'geo': 'IT'}
        print(f"\nTrying to extract data for: {filters}")
        
        # We'll build a time series for all TIME_PERIOD values
        records = []
        
        # Check each value in the dataset
        for pos_str, value in data['value'].items():
            # Parse position
            if ':' in pos_str:
                positions = [int(p) for p in pos_str.split(':')]
            else:
                positions = [int(pos_str)]
            
            # For debugging, show first few positions
            if len(records) < 2:
                print(f"Sample position: {pos_str} -> {positions}")
            
            # Not enough dimensions?
            if len(positions) != len(dimension_order):
                continue
            
            # Check if this matches our filters
            matches = True
            for dim_id, filter_value in filters.items():
                dim_idx = dimension_order.index(dim_id)
                pos_value = positions[dim_idx]
                
                # Find what code this position value corresponds to
                dim_data = data['dimension'].get(dim_id, {})
                category = dim_data.get('category', {})
                index_map = category.get('index', {})
                
                # We need to find the key that has this index value
                found_key = None
                for key, idx in index_map.items():
                    if idx == pos_value:
                        found_key = key
                        break
                
                if found_key != filter_value:
                    matches = False
                    break
            
            if matches:
                # Get TIME_PERIOD value
                time_idx = positions[time_pos]
                time_data = data['dimension'].get('TIME_PERIOD', {})
                time_category = time_data.get('category', {})
                time_index = time_category.get('index', {})
                time_labels = time_category.get('label', {})
                
                # Find time code
                time_code = None
                for code, idx in time_index.items():
                    if idx == time_idx:
                        time_code = code
                        break
                
                time_label = time_labels.get(time_code, f"Period_{time_idx}")
                
                records.append({
                    'period': time_label,
                    'value': value
                })
                
                # Debug output for first few records
                if len(records) <= 2:
                    print(f"Found match: period={time_label}, value={value}")
        
        if records:
            print(f"\nExtracted {len(records)} data points")
            
            # Create DataFrame
            df = pd.DataFrame(records)
            df.set_index('period', inplace=True)
            
            # Show sample
            print("\nSample data:")
            print(df.head())
            
            # Export
            export_path = exports_dir / f"{dataset_id}_sample.csv"
            df.to_csv(export_path)
            print(f"\nExported to {export_path}")
        else:
            print("\nNo data found for the specified filters")
            
            # Try to analyze why no data was found
            print("\nAnalyzing possible issues:")
            
            # Check a few values to see what's in them
            print("\nSampling some data values:")
            sample_count = 0
            for pos_str, value in data['value'].items():
                if sample_count >= 5:
                    break
                    
                # Parse position
                if ':' in pos_str:
                    positions = [int(p) for p in pos_str.split(':')]
                else:
                    positions = [int(pos_str)]
                
                if len(positions) != len(dimension_order):
                    continue
                
                # Get dimension values for this position
                dim_values = {}
                for i, dim_id in enumerate(dimension_order):
                    pos_value = positions[i]
                    dim_data = data['dimension'].get(dim_id, {})
                    category = dim_data.get('category', {})
                    index_map = category.get('index', {})
                    
                    # Find key for this position
                    found_key = None
                    for key, idx in index_map.items():
                        if idx == pos_value:
                            found_key = key
                            break
                    
                    dim_values[dim_id] = found_key
                
                print(f"Position {pos_str}: {dim_values} -> value={value}")
                sample_count += 1

# Run the extraction
simple_extract('AACT_ALI01')