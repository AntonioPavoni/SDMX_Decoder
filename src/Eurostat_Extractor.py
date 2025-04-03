#!/usr/bin/env python3
"""
Enhanced Eurostat time series extractor with multi-dataset support and resume functionality
"""

import os
import json
import gzip
import pandas as pd
from pathlib import Path
import sys
import time
import itertools
import argparse
import traceback

# Base paths
data_dir = Path("C:/Users/Antonio/sdmx/eurostat_sdmx/data")
cache_dir = data_dir / "cache"
datasets_dir = data_dir / "datasets"
exports_dir = data_dir / "exports"
progress_file = data_dir / "time_series_progress.json"

# Create exports directory if it doesn't exist
exports_dir.mkdir(exist_ok=True, parents=True)

# Debug info
print(f"Looking for datasets in:")
print(f"Cache directory: {cache_dir} (exists: {cache_dir.exists()})")
print(f"Metadata directory: {datasets_dir} (exists: {datasets_dir.exists()})")

def extract_time_series(dataset_id, filters=None, debug=True):
    """Extract time series from a dataset with flat indexes and export as JSON"""
    if debug:
        print(f"Extracting time series for {dataset_id}")
    
    # Load metadata
    metadata_file = datasets_dir / f"{dataset_id}_metadata.json"
    if not metadata_file.exists():
        if debug:
            print(f"Metadata file for {dataset_id} not found at {metadata_file}")
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        if debug:
            print(f"Error loading metadata for {dataset_id}: {e}")
        return None
    
    # Load cached data - try multiple file formats
    data_file = cache_dir / f"{dataset_id}_data.json.gz"
    if not data_file.exists():
        # Try alternative file names
        alternatives = [
            cache_dir / f"{dataset_id}.json.gz",
            cache_dir / f"{dataset_id}_data.json",
            cache_dir / f"{dataset_id}.json"
        ]
        for alt in alternatives:
            if alt.exists():
                data_file = alt
                break
    
    if not data_file.exists():
        if debug:
            print(f"Cached data for {dataset_id} not found. Tried multiple file patterns.")
        return None
    
    # Load the data file
    if debug:
        print(f"Loading cached data from {data_file}")
    
    try:
        if data_file.suffix == '.gz':
            with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                if debug:
                    print("Successfully loaded data with gzip.open")
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if debug:
                    print("Successfully loaded data as JSON")
    except Exception as e:
        if debug:
            print(f"Error loading data: {e}")
            print("Trying fallback methods...")
        
        try:
            with open(data_file, 'rb') as f:
                compressed = f.read()
                
            # Try to decompress if it might be gzipped
            if data_file.suffix == '.gz' or compressed[:2] == b'\x1f\x8b':
                decompressed = gzip.decompress(compressed)
                text = decompressed.decode('utf-8')
            else:
                text = compressed.decode('utf-8')
                
            data = json.loads(text)
            if debug:
                print("Successfully loaded data with fallback method")
        except Exception as e2:
            if debug:
                print(f"All loading methods failed: {e2}")
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
            print(f"Dimension order: {dimension_order}")
            print(f"Dimension sizes: {dimension_sizes}")
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
                
            time_values[idx] = {"code": code, "label": label}
    
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
    for time_idx, time_info in time_values.items():
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
                "time_code": time_info["code"],
                "time_label": time_info["label"],
                "value": value
            })
    
    # Create JSON output
    if results:
        # Sort results by time code (try numeric sort first, fall back to string sort)
        try:
            results.sort(key=lambda x: int(x["time_code"]))
        except:
            results.sort(key=lambda x: x["time_code"])
        
        # Create filter info with labels
        filter_info = []
        for dim_id, filter_value in filters.items():
            if dim_id in meta_dimensions:
                dim_info = meta_dimensions[dim_id]
                dim_values = dim_info.get('values', {})
                label = dim_values.get(filter_value, filter_value)
                
                filter_info.append({
                    "id": dim_id,
                    "position": dim_info.get('position'),
                    "values": {
                        filter_value: label
                    }
                })
        
        # Create the JSON output structure
        output = {
            "id": dataset_id,
            "title": metadata.get("title", ""),
            "dimensions": filter_info,
            "time_dimension": {
                "id": time_dim_meta,
                "position": meta_dimensions.get(time_dim_meta, {}).get('position')
            },
            "observations": []
        }
        
        # Add the data points
        for result in results:
            output["observations"].append({
                "time": result["time_label"],
                "value": result["value"]
            })
        
        if debug:
            print(f"\nExtracted {len(results)} data points")
        
        # Export to JSON
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filter_str = "_".join([f"{k}_{v}" for k, v in filters.items()])
        export_path = exports_dir / f"{dataset_id}_{filter_str}_{timestamp}.json"
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        if debug:
            print(f"\nExported to {export_path}")
        
        return output
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

def extract_time_series_for_dataset(dataset_id, limit=None):
    """Extract all possible time series from a dataset"""
    print(f"Extracting all time series for {dataset_id}")
    
    # Load metadata
    metadata_file = datasets_dir / f"{dataset_id}_metadata.json"
    if not metadata_file.exists():
        print(f"Metadata file for {dataset_id} not found at {metadata_file}")
        return False
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata for {dataset_id}: {e}")
        return False
    
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
        print(f"Error: TIME_PERIOD dimension not found for {dataset_id}")
        return False
    
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
    
    # Apply limit if specified
    if limit and limit < len(combinations):
        print(f"Limiting to first {limit} combinations")
        combinations = combinations[:limit]
    
    # Extract time series for each combination
    successful = 0
    for i, filters in enumerate(combinations):
        print(f"\nProcessing combination {i+1}/{len(combinations)}: {filters}")
        # Turn off detailed debugging for all except first few
        debug_mode = (i < 3 or successful < 2)
        
        try:
            output = extract_time_series(dataset_id, filters, debug=debug_mode)
            if output is not None:
                successful += 1
        except Exception as e:
            print(f"Error processing combination {filters}: {e}")
            if debug_mode:
                traceback.print_exc()
    
    print(f"\nExtracted {successful} time series out of {len(combinations)} combinations")
    return successful > 0

def get_datasets_from_cache():
    """Get all dataset IDs from cache directory"""
    datasets = []
    
    # Look for files with _data.json.gz pattern
    for file in cache_dir.glob("*_data.json.gz"):
        dataset_id = file.stem.replace("_data.json", "")
        datasets.append(dataset_id)
    
   
    return datasets

def get_datasets_from_metadata():
    """Get all dataset IDs from metadata directory"""
    datasets = []
    for file in datasets_dir.glob("*_metadata.json"):
        dataset_id = file.stem.replace("_metadata", "")
        datasets.append(dataset_id)
    return datasets

def load_progress():
    """Load progress from progress file"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(progress):
    """Save progress to progress file"""
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def extract_all_datasets(start_index=0, limit=None, dataset_limit=None):
    """Extract time series for all datasets"""
    # Get all datasets from cache and metadata
    cache_datasets = set(get_datasets_from_cache())
    metadata_datasets = set(get_datasets_from_metadata())
    
    print(f"Found {len(cache_datasets)} datasets in cache")
    print(f"Found {len(metadata_datasets)} datasets in metadata")
    
    # Only process datasets that have both cache and metadata
    datasets = sorted(list(cache_datasets.intersection(metadata_datasets)))
    
    if not datasets:
        print("No datasets found with both cache and metadata")
        return
    
    print(f"Found {len(datasets)} datasets with both cache and metadata")
    
    # Load progress
    progress = load_progress()
    
    # Apply dataset limit if specified
    if dataset_limit and dataset_limit < len(datasets):
        datasets = datasets[:dataset_limit]
        print(f"Limiting to first {dataset_limit} datasets")
    
    # Start from the specified index
    if start_index > 0:
        if start_index >= len(datasets):
            print(f"Start index {start_index} is out of range (max: {len(datasets) - 1})")
            return
        
        print(f"Starting from dataset {start_index}: {datasets[start_index]}")
        datasets = datasets[start_index:]
    
    # Process each dataset
    for i, dataset_id in enumerate(datasets):
        overall_index = start_index + i
        print(f"\n[{overall_index + 1}/{len(datasets) + start_index}] Processing dataset: {dataset_id}")
        
        # Skip if already processed successfully
        if dataset_id in progress and progress[dataset_id].get('status') == 'success':
            print(f"Dataset {dataset_id} already processed successfully. Skipping.")
            continue
        
        # Process the dataset
        start_time = time.time()
        
        try:
            success = extract_time_series_for_dataset(dataset_id, limit)
            
            # Update progress
            progress[dataset_id] = {
                'status': 'success' if success else 'failure',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration': round(time.time() - start_time, 2)
            }
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")
            traceback.print_exc()
            
            # Update progress
            progress[dataset_id] = {
                'status': 'error',
                'error': str(e),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration': round(time.time() - start_time, 2)
            }
        
        # Save progress after each dataset
        save_progress(progress)
        
        print(f"Processed {dataset_id} in {round(time.time() - start_time, 2)} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract time series from Eurostat datasets')
    parser.add_argument('--dataset', help='Process a specific dataset ID')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    parser.add_argument('--start', type=int, default=0, help='Start index for processing datasets')
    parser.add_argument('--limit', type=int, help='Limit number of combinations per dataset')
    parser.add_argument('--dataset-limit', type=int, help='Limit number of datasets to process')
    parser.add_argument('--list', action='store_true', help='List all available datasets')
    parser.add_argument('--progress', action='store_true', help='Show processing progress')
    
    args = parser.parse_args()
    
    if args.list:
        # Get all datasets from cache and metadata
        cache_datasets = set(get_datasets_from_cache())
        metadata_datasets = set(get_datasets_from_metadata())
        
        # Show datasets with both cache and metadata
        both = sorted(list(cache_datasets.intersection(metadata_datasets)))
        print(f"\nDatasets with both cache and metadata ({len(both)}):")
        for i, dataset_id in enumerate(both):
            print(f"{i}: {dataset_id}")
        
        # Show datasets with only cache
        cache_only = sorted(list(cache_datasets - metadata_datasets))
        if cache_only:
            print(f"\nDatasets with cache only ({len(cache_only)}):")
            for dataset_id in cache_only:
                print(f"  {dataset_id}")
        
        # Show datasets with only metadata
        metadata_only = sorted(list(metadata_datasets - cache_datasets))
        if metadata_only:
            print(f"\nDatasets with metadata only ({len(metadata_only)}):")
            for dataset_id in metadata_only:
                print(f"  {dataset_id}")
        
        return
    
    if args.progress:
        # Show progress
        progress = load_progress()
        
        if not progress:
            print("No progress data found")
            return
        
        # Count by status
        status_counts = {}
        for dataset_id, info in progress.items():
            status = info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nProcessing progress:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # List completed datasets
        successful = [dataset_id for dataset_id, info in progress.items() if info.get('status') == 'success']
        if successful:
            print(f"\nSuccessfully processed datasets ({len(successful)}):")
            for dataset_id in sorted(successful):
                print(f"  {dataset_id}")
        
        # List failed datasets
        failed = [dataset_id for dataset_id, info in progress.items() if info.get('status') in ('failure', 'error')]
        if failed:
            print(f"\nFailed datasets ({len(failed)}):")
            for dataset_id in sorted(failed):
                info = progress[dataset_id]
                error = info.get('error', 'unknown error')
                print(f"  {dataset_id}: {error}")
        
        return
    
    if args.dataset:
        # Process a specific dataset
        extract_time_series_for_dataset(args.dataset, args.limit)
    elif args.all:
        # Process all datasets
        extract_all_datasets(args.start, args.limit, args.dataset_limit)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()