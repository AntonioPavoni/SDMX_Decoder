#!/usr/bin/env python3
"""
TimeSeriesExtraction.py - Extract time series from Eurostat SDMX cached data

This script extracts time series from cached Eurostat SDMX data based on dataset metadata.
It can process all possible combinations of dimension values (except time dimension) to create
multiple time series for a single dataset.
"""

import os
import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Optional

# Import the improved TimeSeriesExtractor class
from time_series_extractor import TimeSeriesExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract time series from cached Eurostat datasets',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset', type=str, help='Dataset ID to process')
    parser.add_argument('--data-dir', type=str, help='Base data directory path')
    parser.add_argument('--list-series', action='store_true', 
                      help='List available time series for the dataset')
    parser.add_argument('--export-all', action='store_true',
                      help='Extract and export all possible time series for the dataset')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit number of time series to extract (default: all)')
    parser.add_argument('--filter', type=str, nargs='*', 
                      help='Filters in format dim=value (e.g., --filter freq=A itm_newa=42000)')
    parser.add_argument('--export-format', type=str, choices=['csv', 'excel', 'json'], default='csv',
                      help='Export format for time series (default: csv)')
    parser.add_argument('--verbose', action='store_true', 
                      help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug level logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Check if dataset ID is provided
    if not args.dataset:
        print("Error: Please specify a dataset ID using --dataset")
        parser.print_help()
        return
    
    # Initialize the extractor
    extractor = TimeSeriesExtractor(base_dir=args.data_dir)
    
    # Convert filter arguments to a dictionary
    filters = {}
    if args.filter:
        for filter_str in args.filter:
            parts = filter_str.split('=', 1)  # Split on first = only
            if len(parts) == 2:
                filters[parts[0]] = parts[1]
            else:
                print(f"Warning: Invalid filter format '{filter_str}'. Skipping.")
    
    try:
        # Get metadata to display information about the dataset
        metadata = extractor.load_dataset_metadata(args.dataset)
        if metadata:
            print(f"\nDataset: {args.dataset} - {metadata.get('title', 'Unknown')}")
            
            # Get dimensions from metadata
            dimensions = metadata.get('dimensions', [])
            if dimensions:
                print("\nDimensions:")
                for dim in dimensions:
                    dim_id = dim.get('id', 'Unknown')
                    position = dim.get('position', 'Unknown')
                    values = dim.get('values', {})
                    
                    print(f"  Position {position}: {dim_id} ({len(values)} values)")
                    
                    # If verbose, show the actual values
                    if args.verbose and values:
                        print("    Values:")
                        for code, label in list(values.items())[:5]:  # Show first 5 values
                            print(f"      {code}: {label}")
                        
                        if len(values) > 5:
                            print(f"      ... and {len(values) - 5} more values")
        else:
            print(f"Error: Could not load metadata for dataset {args.dataset}")
            return
        
        # List available time series configurations
        if args.list_series:
            # Get dimensions by position
            dimensions_by_position = extractor.get_dimensions_by_position(metadata)
            
            # Find time dimension
            time_position = extractor.find_time_dimension(dimensions_by_position)
            if time_position is not None:
                print(f"\nTime dimension identified at position {time_position}")
                
                # Generate filter combinations
                filter_combinations = extractor.generate_filter_combinations(dimensions_by_position, time_position)
                
                print(f"\nFound {len(filter_combinations)} possible time series configurations")
                
                # Show the first few combinations as examples
                max_to_show = min(5, len(filter_combinations))
                for i in range(max_to_show):
                    filters = filter_combinations[i]
                    
                    # Create a descriptive string
                    description = []
                    for dim_id, value in filters.items():
                        # Find dimension in metadata
                        for dim in dimensions:
                            if dim.get('id') == dim_id:
                                label = dim.get('values', {}).get(value, value)
                                description.append(f"{dim_id}={label}")
                                break
                    
                    print(f"\nConfiguration #{i+1}:")
                    print(f"  Filters: {filters}")
                    print(f"  Description: {' AND '.join(description)}")
                    
                    # Show command to extract this specific configuration
                    filter_args = " ".join([f"--filter {k}={v}" for k, v in filters.items()])
                    print(f"\n  Example command:")
                    print(f"  python timeSeriesExtraction.py --dataset {args.dataset} {filter_args}")
                
                if len(filter_combinations) > max_to_show:
                    print(f"\n...and {len(filter_combinations) - max_to_show} more configurations")
                    
                # Show command to extract all configurations
                print("\nTo extract all configurations:")
                print(f"  python timeSeriesExtraction.py --dataset {args.dataset} --export-all")
            else:
                print("Error: Could not identify time dimension in metadata")
        
        # Extract and export all possible time series
        elif args.export_all:
            print(f"\nExtracting all time series for dataset {args.dataset}...")
            if args.limit:
                print(f"  (limiting to {args.limit} series)")
                
            # Load cached data
            cached_data = extractor.load_cached_data(args.dataset)
            if not cached_data:
                print(f"Error: Could not load cached data for dataset {args.dataset}")
                return
                
            # Get dimensions by position
            dimensions_by_position = extractor.get_dimensions_by_position(metadata)
            
            # Find time dimension
            time_position = extractor.find_time_dimension(dimensions_by_position)
            if time_position is None:
                print("Error: Could not identify time dimension in metadata")
                return
                
            # Generate filter combinations
            filter_combinations = extractor.generate_filter_combinations(dimensions_by_position, time_position)
            
            if args.limit and args.limit < len(filter_combinations):
                filter_combinations = filter_combinations[:args.limit]
                
            # Process each combination
            time_series_list = []
            print(f"\nProcessing {len(filter_combinations)} combinations...")
            
            # Get dimension indices from cached data
            dimension_indices = extractor._get_dimension_indices(metadata, cached_data)
            if not dimension_indices:
                print("Error: Could not extract dimension indices from metadata and cached data")
                return
                
            for i, filters in enumerate(filter_combinations):
                print(f"  Processing combination {i+1}/{len(filter_combinations)}: {filters}")
                
                # Create time series
                df = extractor.create_time_series_improved(args.dataset, cached_data, metadata, filters)
                
                if df is not None and not df.empty:
                    time_series_list.append(df)
                
                # Progress update every 10 combinations
                if (i+1) % 10 == 0 or i+1 == len(filter_combinations):
                    print(f"  Processed {i+1}/{len(filter_combinations)} combinations, found {len(time_series_list)} valid series")
            
            # Export all time series
            if time_series_list:
                print(f"\nSuccessfully extracted {len(time_series_list)} time series")
                
                export_paths = extractor.export_all_time_series(time_series_list, args.export_format)
                
                if export_paths:
                    print(f"\nExported {len(export_paths)} time series to:")
                    for path in export_paths[:5]:  # Show first 5 paths
                        print(f"  {path}")
                        
                    if len(export_paths) > 5:
                        print(f"  ... and {len(export_paths) - 5} more files")
                else:
                    print("Error: Failed to export time series")
            else:
                print("No time series could be extracted. Please check your data.")
        
        # Extract a single time series with specified filters
        elif filters:
            print(f"\nExtracting time series for dataset {args.dataset} with filters:")
            for dim_id, value in filters.items():
                # Find dimension in metadata
                for dim in dimensions:
                    if dim.get('id') == dim_id:
                        label = dim.get('values', {}).get(value, value)
                        print(f"  {dim_id} = {label}")
                        break
            
            # Load cached data
            cached_data = extractor.load_cached_data(args.dataset)
            if cached_data:
                # Create time series
                df = extractor.create_time_series_improved(args.dataset, cached_data, metadata, filters)
                
                if df is not None and not df.empty:
                    print(f"\nSuccessfully extracted time series with {len(df)} data points")
                    
                    # Show sample data
                    print("\nSample data:")
                    pd.set_option('display.max_rows', 10)
                    pd.set_option('display.width', 120)
                    print(df)
                    
                    # Export time series
                    filter_hash = "_".join([f"{k}_{v}" for k, v in filters.items()])
                    export_path = extractor.export_time_series(df, filter_hash, args.export_format)
                    
                    if export_path:
                        print(f"\nExported time series to: {export_path}")
                    else:
                        print("Error: Failed to export time series")
                else:
                    print("No data found for the specified filters")
            else:
                print(f"Error: Could not load cached data for dataset {args.dataset}")
        
        # No action specified
        else:
            print("\nNo action specified. Please use one of the following:")
            print("  --list-series : List available time series configurations")
            print("  --export-all : Extract and export all time series")
            print("  --filter dim=value : Extract a specific time series")
            parser.print_help()
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if args.debug:
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")