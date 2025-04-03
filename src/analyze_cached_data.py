import os
import json
import gzip
import sys
import argparse
from pathlib import Path
from pprint import pprint
import pandas as pd

# Ensure the script can find modules in the project
current_dir = Path(__file__).parent
root_dir = current_dir.parent  # This would be the EUROSTAT_SDMX directory
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

def load_cached_data(dataset_id, base_dir=None):
    """Load the cached data for the specified dataset ID"""
    if base_dir is None:
        base_dir = root_dir / "data"
    
    cache_dir = Path(base_dir) / "cache"
    cache_path = cache_dir / f"{dataset_id}_data.json.gz"
    
    if not cache_path.exists():
        print(f"Error: Cached data not found at {cache_path}")
        return None
    
    try:
        # Try different approaches to read the gzipped JSON file
        try:
            # First attempt: standard gzip opening
            with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"First attempt failed: {e}")
            
            # Second attempt: manual decompression
            with open(cache_path, 'rb') as f:
                compressed_data = f.read()
            
            decompressed_data = gzip.decompress(compressed_data)
            return json.loads(decompressed_data.decode('utf-8'))
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None

def analyze_dataset_structure(data):
    """Analyze and print the structure of the dataset"""
    if not data:
        print("No data to analyze")
        return
    
    print("\n" + "="*80)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*80)
    
    # Print top-level keys
    print("\nTop-level keys in the dataset:")
    for key in data.keys():
        print(f"  - {key}")
    
    # Analyze dimension structure
    if "dimension" in data:
        dimensions = data["dimension"]
        print(f"\nFound {len(dimensions)} dimensions:")
        
        for dim_name, dim_data in dimensions.items():
            print(f"\n  Dimension: {dim_name}")
            
            # Print dimension keys
            print(f"    Keys: {', '.join(dim_data.keys())}")
            
            # If there's a category, examine it
            if "category" in dim_data:
                category = dim_data["category"]
                print(f"    Category keys: {', '.join(category.keys())}")
                
                # Check for index structure
                if "index" in category:
                    index = category["index"]
                    index_type = type(index).__name__
                    print(f"    Index type: {index_type}")
                    
                    if isinstance(index, dict):
                        print(f"    Index entries: {len(index)}")
                        
                        # Print a few sample entries
                        sample_indices = list(index.items())[:5]
                        print(f"    Sample indices: {sample_indices}")
                    elif isinstance(index, list):
                        print(f"    Index entries: {len(index)}")
                        print(f"    Sample indices: {index[:5]}")
                
                # Check for label structure
                if "label" in category:
                    labels = category["label"]
                    print(f"    Label entries: {len(labels)}")
                    
                    # Print a few sample labels
                    sample_labels = list(labels.items())[:5]
                    print(f"    Sample labels: {sample_labels}")
    
    # Analyze value structure
    if "value" in data:
        values = data["value"]
        value_count = len(values)
        print(f"\nFound {value_count} data values.")
        
        # Print a few sample values
        sample_values = list(values.items())[:5]
        print("\nSample values (position -> value):")
        for position, value in sample_values:
            print(f"  {position} -> {value}")
    
    # Analyze dimension order and size
    if "id" in data:
        dimension_order = data["id"]
        print(f"\nDimension order: {dimension_order}")
    
    if "size" in data:
        dimension_sizes = data["size"]
        print(f"\nDimension sizes: {dimension_sizes}")
        
        # If we have both order and sizes, create a mapping
        if "id" in data:
            size_mapping = {data["id"][i]: size for i, size in enumerate(dimension_sizes)}
            print("\nDimension size mapping:")
            for dim, size in size_mapping.items():
                print(f"  {dim}: {size}")

def find_time_dimension(data):
    """Find the time dimension in the dataset"""
    if "id" not in data:
        return None, -1
    
    dimension_order = data["id"]
    
    for i, dim in enumerate(dimension_order):
        if dim.lower() in ('time_period', 'time'):
            return dim, i
    
    # If no explicit time dimension, use the last one as a fallback
    return dimension_order[-1], len(dimension_order) - 1

def analyze_position_format(data):
    """Analyze the format of position strings in the dataset"""
    if "value" not in data:
        return
    
    print("\n" + "="*80)
    print("POSITION FORMAT ANALYSIS")
    print("="*80)
    
    positions = list(data["value"].keys())
    
    if not positions:
        print("\nNo positions found in the dataset")
        return
    
    # Check the format of the first position
    first_pos = positions[0]
    print(f"\nFirst position string: '{first_pos}'")
    
    if ':' in first_pos:
        parts = first_pos.split(':')
        print(f"This is a colon-separated position with {len(parts)} parts: {parts}")
        
        # Check if the number of parts matches the number of dimensions
        if "id" in data:
            dimension_order = data["id"]
            if len(parts) == len(dimension_order):
                print("The number of parts matches the number of dimensions.")
                
                # Map each part to its dimension
                print("\nMapping position parts to dimensions:")
                for i, (dim, part) in enumerate(zip(dimension_order, parts)):
                    print(f"  Part {i}: {part} (Dimension: {dim})")
            else:
                print(f"Warning: Position has {len(parts)} parts but there are {len(dimension_order)} dimensions.")
    else:
        print("This is a single-value position string.")
        
        # Check if this is a single-dimension dataset
        if "id" in data and len(data["id"]) == 1:
            print(f"This matches the single dimension: {data['id'][0]}")
    
    # Check if all positions have the same format
    all_same_format = all((':' in pos) == (':' in first_pos) for pos in positions)
    all_same_parts = all(len(pos.split(':')) == len(first_pos.split(':')) for pos in positions if ':' in pos and ':' in first_pos)
    
    if all_same_format and (not ':' in first_pos or all_same_parts):
        print("\nAll positions have the same format.")
    else:
        print("\nWarning: Positions have inconsistent formats.")
        
        # Count positions with different part counts
        if ':' in first_pos:
            part_counts = {}
            for pos in positions:
                if ':' in pos:
                    count = len(pos.split(':'))
                    part_counts[count] = part_counts.get(count, 0) + 1
                else:
                    part_counts['single'] = part_counts.get('single', 0) + 1
            
            print("Position format distribution:")
            for count, num in part_counts.items():
                if count == 'single':
                    print(f"  Single-value positions: {num}")
                else:
                    print(f"  Positions with {count} parts: {num}")

def find_data_points_for_dimension_values(data, dimensions_to_check=None):
    """Find how many data points exist for each value of specified dimensions"""
    if "value" not in data or "dimension" not in data or "id" not in data:
        return
    
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    dimension_order = data["id"]
    time_dim, time_idx = find_time_dimension(data)
    
    # If no dimensions specified, analyze all dimensions except time
    if not dimensions_to_check:
        dimensions_to_check = [dim for i, dim in enumerate(dimension_order) if i != time_idx]
    
    for dim_name in dimensions_to_check:
        if dim_name not in data["dimension"]:
            continue
            
        dim_data = data["dimension"][dim_name]
        if "category" not in dim_data or "label" not in dim_data["category"]:
            continue
            
        dim_idx = dimension_order.index(dim_name)
        labels = dim_data["category"]["label"]
        
        print(f"\nData distribution for dimension: {dim_name}")
        
        # Count data points for each value of this dimension
        value_counts = {}
        
        for position, value in data["value"].items():
            parts = position.split(':')
            
            # Skip positions with wrong format
            if len(parts) != len(dimension_order):
                continue
                
            part_index = int(parts[dim_idx])
            
            # Find the corresponding label
            dim_value = None
            if "index" in dim_data["category"]:
                index_data = dim_data["category"]["index"]
                
                if isinstance(index_data, dict):
                    # Find the key with this index
                    for key, idx in index_data.items():
                        if idx == part_index:
                            dim_value = key
                            break
                elif isinstance(index_data, list) and part_index < len(index_data):
                    dim_value = index_data[part_index]
            
            if dim_value is None:
                dim_value = f"Value_{part_index}"
                
            # Get the label
            label = labels.get(dim_value, dim_value)
            
            # Count this occurrence
            key = f"{dim_value} ({label})"
            value_counts[key] = value_counts.get(key, 0) + 1
        
        # Print the counts
        print(f"  Found {len(value_counts)} distinct values with data:")
        
        # Sort by count, descending
        sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        for value, count in sorted_counts:
            print(f"    {value}: {count} data points")

def extract_sample_time_series(data, filters=None):
    """Extract a sample time series using the specified filters"""
    if "value" not in data or "dimension" not in data or "id" not in data:
        return
    
    print("\n" + "="*80)
    print("SAMPLE TIME SERIES EXTRACTION")
    print("="*80)
    
    dimension_order = data["id"]
    time_dim, time_idx = find_time_dimension(data)
    
    print(f"\nTime dimension identified as: {time_dim} (index {time_idx})")
    
    # If no filters provided, create default filters
    if not filters:
        filters = {}
        
        # Use the first value of each non-time dimension
        for i, dim_name in enumerate(dimension_order):
            if i == time_idx:
                continue
                
            dim_data = data["dimension"].get(dim_name)
            if not dim_data or "category" not in dim_data:
                continue
                
            # Find the first category value
            if "index" in dim_data["category"]:
                index_data = dim_data["category"]["index"]
                
                if isinstance(index_data, dict):
                    # Get the key with the lowest index
                    min_idx = min(index_data.values())
                    for key, idx in index_data.items():
                        if idx == min_idx:
                            filters[dim_name] = key
                            break
                elif isinstance(index_data, list) and index_data:
                    filters[dim_name] = index_data[0]
    
    print(f"\nUsing filters: {filters}")
    
    # Collect the time series data points
    time_series = []
    
    for position, value in data["value"].items():
        parts = position.split(':')
        
        # Skip positions with wrong format
        if len(parts) != len(dimension_order):
            continue
            
        position_indices = list(map(int, parts))
        
        # Check if this position matches our filters
        matches = True
        for i, dim_name in enumerate(dimension_order):
            if i == time_idx:
                continue  # Skip time dimension
                
            if dim_name in filters:
                dim = data["dimension"].get(dim_name)
                if not dim or "category" not in dim or "index" not in dim["category"]:
                    continue
                    
                # Get the expected index for this filter value
                filter_value = filters[dim_name]
                expected_index = None
                
                if isinstance(dim["category"]["index"], dict):
                    expected_index = dim["category"]["index"].get(filter_value)
                elif isinstance(dim["category"]["index"], list):
                    try:
                        expected_index = dim["category"]["index"].index(filter_value)
                    except ValueError:
                        pass
                
                # If we couldn't find an index or it doesn't match, skip this position
                if expected_index is None or position_indices[i] != expected_index:
                    matches = False
                    break
        
        if not matches:
            continue
            
        # Get the time value
        time_index = position_indices[time_idx]
        time_dim_data = data["dimension"].get(time_dim)
        
        if not time_dim_data or "category" not in time_dim_data:
            continue
            
        # Get the time period label
        time_value = None
        if "index" in time_dim_data["category"] and "label" in time_dim_data["category"]:
            if isinstance(time_dim_data["category"]["index"], dict):
                # Find the key with this index
                for key, idx in time_dim_data["category"]["index"].items():
                    if idx == time_index:
                        time_value = time_dim_data["category"]["label"].get(key)
                        break
            elif isinstance(time_dim_data["category"]["index"], list):
                if time_index < len(time_dim_data["category"]["index"]):
                    time_key = time_dim_data["category"]["index"][time_index]
                    time_value = time_dim_data["category"]["label"].get(time_key)
        
        if time_value is None:
            time_value = f"Period_{time_index}"
            
        # Add to time series
        time_series.append({
            "time": time_value,
            "value": value
        })
    
    # Sort by time
    time_series.sort(key=lambda x: x["time"])
    
    # Print the time series
    if time_series:
        print(f"\nExtracted time series with {len(time_series)} data points:")
        df = pd.DataFrame(time_series)
        print(df)
    else:
        print("\nNo data points found for the specified filters")

def main():
    parser = argparse.ArgumentParser(description='Analyze cached Eurostat dataset')
    parser.add_argument('dataset_id', type=str, help='Dataset ID to analyze')
    parser.add_argument('--data-dir', type=str, help='Base data directory path')
    parser.add_argument('--filter', type=str, nargs='*', help='Filters in format dim=value for sample extraction')
    
    args = parser.parse_args()
    
    # Load the cached data
    data = load_cached_data(args.dataset_id, args.data_dir)
    
    if not data:
        print(f"Failed to load cached data for {args.dataset_id}")
        return
    
    print(f"\nSuccessfully loaded cached data for {args.dataset_id}")
    
    # Convert filter arguments to a dictionary
    filters = {}
    if args.filter:
        for filter_str in args.filter:
            parts = filter_str.split('=')
            if len(parts) == 2:
                filters[parts[0]] = parts[1]
    
    # Analyze the data structure
    analyze_dataset_structure(data)
    
    # Analyze position format
    analyze_position_format(data)
    
    # Analyze data distribution for dimensions
    find_data_points_for_dimension_values(data)
    
    # Extract a sample time series
    extract_sample_time_series(data, filters)

if __name__ == "__main__":
    main()