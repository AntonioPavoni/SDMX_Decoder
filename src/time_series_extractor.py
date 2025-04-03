import os
import json
import gzip
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
import itertools
from typing import Dict, List, Optional, Union, Tuple, Any

class TimeSeriesExtractor:
    def __init__(self, base_dir: str = None):
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent  # This would be the EUROSTAT_SDMX directory
        
        if base_dir is None:
            # According to your directory structure, data directory should be at root_dir/data
            data_path = root_dir / "data"
            
            if data_path.exists() and data_path.is_dir():
                self.base_dir = data_path
                logging.info(f"Using data directory at: {data_path}")
            else:
                self.base_dir = data_path
                logging.warning(f"Data directory not found at expected location: {data_path}")
                logging.warning("Creating directory structure...")
                data_path.mkdir(exist_ok=True, parents=True)
        else:
            self.base_dir = Path(base_dir)
            
        # Set up directory paths based on the project structure
        self.datasets_dir = self.base_dir / "datasets"
        self.cache_dir = self.base_dir / "cache"
        self.exports_dir = self.base_dir / "exports"
        
        # Create directories if they don't exist
        self.exports_dir.mkdir(exist_ok=True, parents=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_path = Path(__file__).parent / "time_series_extraction.log"
        
        # Ensure the log directory exists
        log_path.parent.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_path
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        logging.info(f"Base data directory: {self.base_dir}")
        logging.info(f"Datasets directory: {self.datasets_dir}")
        logging.info(f"Cache directory: {self.cache_dir}")
        logging.info(f"Exports directory: {self.exports_dir}")
    
    def load_dataset_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Load the metadata file for a dataset"""
        metadata_file = self.datasets_dir / f"{dataset_id}_metadata.json"
        if not metadata_file.exists():
            logging.error(f"Metadata file for {dataset_id} not found at {metadata_file}")
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading metadata for {dataset_id}: {e}")
            return None
    
    def load_cached_data(self, dataset_id: str) -> Optional[Dict]:
        """Load cached data for a dataset with improved error handling"""
        cache_path = self.cache_dir / f"{dataset_id}_data.json.gz"
        if not cache_path.exists():
            logging.error(f"Cached data for {dataset_id} not found at {cache_path}")
            return None
        
        try:
            # Try multiple approaches to read the gzipped JSON
            
            # Approach 1: Use gzip.open directly
            try:
                with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Successfully loaded cached data using gzip.open")
                    return data
            except Exception as e1:
                logging.warning(f"Error reading with gzip.open: {e1}")
            
            # Approach 2: Read as bytes and decompress manually
            try:
                with open(cache_path, 'rb') as f:
                    compressed_data = f.read()
                
                # Try to decompress
                try:
                    decompressed_data = gzip.decompress(compressed_data)
                    json_str = decompressed_data.decode('utf-8')
                    data = json.loads(json_str)
                    logging.info(f"Successfully loaded cached data using manual decompression")
                    return data
                except Exception as e2:
                    logging.warning(f"Error in manual decompression: {e2}")
            except Exception as e3:
                logging.warning(f"Error reading file as binary: {e3}")
            
            # Approach 3: Read as binary and try different gzip decompression
            try:
                import zlib
                with open(cache_path, 'rb') as f:
                    compressed_data = f.read()
                
                # Skip gzip header (first 10 bytes) and try zlib.decompress
                # This sometimes works when the gzip header is corrupt
                decompressed_data = zlib.decompress(compressed_data[10:], 15 + 32)
                json_str = decompressed_data.decode('utf-8')
                data = json.loads(json_str)
                logging.info(f"Successfully loaded cached data using zlib with header skip")
                return data
            except Exception as e4:
                logging.warning(f"Error using zlib with header skip: {e4}")
            
            # Approach 4: Try reading as plain JSON (maybe it's not actually compressed)
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Successfully loaded cached data as plain JSON")
                    return data
            except Exception as e5:
                logging.warning(f"Error reading as plain JSON: {e5}")
            
            # If all approaches fail, raise a comprehensive error
            logging.error("All approaches to read the cached data failed")
            return None
            
        except Exception as e:
            logging.error(f"Unexpected error loading cached data for {dataset_id}: {e}")
            return None
    
    def get_dimensions_by_position(self, metadata: Dict) -> Dict[int, Dict]:
        """
        Extract dimensions from metadata sorted by position
        
        Returns:
            Dict[int, Dict]: A dictionary with position as key and dimension info as value
        """
        dimensions_by_position = {}
        
        # Extract dimensions from metadata
        dimensions = metadata.get('dimensions', [])
        for dim in dimensions:
            position = dim.get('position')
            if position is not None:
                dimensions_by_position[position] = {
                    'id': dim.get('id'),
                    'values': dim.get('values', {})
                }
        
        return dimensions_by_position
    
    def find_time_dimension(self, dimensions_by_position: Dict[int, Dict]) -> Optional[int]:
        """Find the position of time dimension in the metadata"""
        for position, dim in dimensions_by_position.items():
            if dim['id'].lower() in ('time_period', 'time'):
                return position
        
        # If no time dimension found, return the highest position (assuming last dimension is time)
        if dimensions_by_position:
            return max(dimensions_by_position.keys())
            
        return None
    
    def generate_filter_combinations(self, dimensions_by_position: Dict[int, Dict], time_position: int) -> List[Dict[str, str]]:
        """
        Generate all possible filter combinations for dimensions (except time dimension)
        
        Args:
            dimensions_by_position: Dictionary of dimensions by position
            time_position: Position of the time dimension
            
        Returns:
            List[Dict[str, str]]: List of filter dictionaries
        """
        dimension_values = []
        dimension_ids = []
        
        # Sort positions to ensure we process in correct order
        positions = sorted(dimensions_by_position.keys())
        
        for position in positions:
            if position != time_position:  # Skip time dimension
                dim = dimensions_by_position[position]
                dimension_ids.append(dim['id'])
                dimension_values.append(list(dim['values'].keys()))
        
        # Generate all combinations
        combinations = list(itertools.product(*dimension_values))
        
        # Convert combinations to filter dictionaries
        filter_combinations = []
        for combo in combinations:
            filter_dict = {}
            for i, dim_id in enumerate(dimension_ids):
                filter_dict[dim_id] = combo[i]
            filter_combinations.append(filter_dict)
        
        return filter_combinations
    
    def extract_all_time_series(self, dataset_id: str, limit: int = None) -> List[pd.DataFrame]:
        """
        Extract all possible time series from a dataset based on dimension combinations
        
        Args:
            dataset_id: Dataset ID to process
            limit: Optional limit on number of time series to extract
            
        Returns:
            List[pd.DataFrame]: List of extracted time series DataFrames
        """
        # Load metadata and cached data
        metadata = self.load_dataset_metadata(dataset_id)
        if not metadata:
            return []
            
        cached_data = self.load_cached_data(dataset_id)
        if not cached_data:
            return []
        
        # Get dimensions by position
        dimensions_by_position = self.get_dimensions_by_position(metadata)
        
        # Find time dimension
        time_position = self.find_time_dimension(dimensions_by_position)
        if time_position is None:
            logging.error(f"Could not identify time dimension for dataset {dataset_id}")
            return []
            
        logging.info(f"Identified time dimension at position {time_position}")
        
        # Generate all filter combinations
        filter_combinations = self.generate_filter_combinations(dimensions_by_position, time_position)
        logging.info(f"Generated {len(filter_combinations)} filter combinations")
        
        # Apply limit if specified
        if limit and limit < len(filter_combinations):
            logging.info(f"Limiting to first {limit} combinations")
            filter_combinations = filter_combinations[:limit]
        
        # Extract time series for each combination
        time_series_list = []
        for i, filters in enumerate(filter_combinations):
            logging.info(f"Processing combination {i+1}/{len(filter_combinations)}: {filters}")
            
            # Create descriptive name for this combination
            combination_desc = []
            for dim_id, value in filters.items():
                dim_position = None
                for pos, dim in dimensions_by_position.items():
                    if dim['id'] == dim_id:
                        dim_position = pos
                        break
                
                if dim_position and value in dimensions_by_position[dim_position]['values']:
                    label = dimensions_by_position[dim_position]['values'][value]
                    combination_desc.append(f"{dim_id}={label}")
                else:
                    combination_desc.append(f"{dim_id}={value}")
            
            combination_name = " AND ".join(combination_desc)
            logging.info(f"Combination description: {combination_name}")
            
            # Extract time series with improved extraction logic
            df = self.create_time_series_improved(dataset_id, cached_data, metadata, filters)
            
            if df is not None and not df.empty:
                # Add combination info to DataFrame attributes
                df.attrs['description'] = combination_name
                df.attrs['filters'] = filters
                
                time_series_list.append(df)
                logging.info(f"Successfully extracted time series with {len(df)} data points")
            else:
                logging.warning(f"No data found for combination: {filters}")
        
        return time_series_list
    
    def create_time_series_improved(self, dataset_id: str, data: Dict, metadata: Dict, filters: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        Create a time series from dataset data with the given filters - improved version
        
        Args:
            dataset_id: Dataset ID
            data: Cached dataset data
            metadata: Dataset metadata
            filters: Filter dictionary
            
        Returns:
            Optional[pd.DataFrame]: Time series as DataFrame or None if no data found
        """
        try:
            # Get dimensions and dimension indices from metadata
            dimension_indices = self._get_dimension_indices(metadata, data)
            if not dimension_indices:
                logging.error("Failed to extract dimension indices from metadata")
                return None
            
            # Get time series using the improved processing function
            time_series = self.process_eurostat_json_improved(data, metadata, filters, dimension_indices)
            
            if time_series.empty:
                return None
                
            # Add metadata attributes
            time_series.attrs['dataset_id'] = dataset_id
            time_series.attrs['title'] = metadata.get('title', 'Unknown')
            time_series.attrs['filters'] = filters
            
            return time_series
            
        except Exception as e:
            logging.error(f"Error creating time series: {e}", exc_info=True)
            return None
    
    def _get_dimension_indices(self, metadata: Dict, data: Dict) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Extract dimension indices from metadata and data
        
        Args:
            metadata: Dataset metadata
            data: Dataset data
            
        Returns:
            Optional[Dict[str, Dict[str, int]]]: Mapping of dimension IDs to value-index pairs
        """
        try:
            dimension_indices = {}
            
            # Get dimensions from metadata
            dimensions = metadata.get('dimensions', [])
            for dim in dimensions:
                dim_id = dim.get('id')
                dim_values = dim.get('values', {})
                
                # Try to get indices from data
                if data.get('dimension') and dim_id in data['dimension']:
                    data_dim = data['dimension'][dim_id]
                    if 'category' in data_dim and 'index' in data_dim['category']:
                        # Data contains indices, use these
                        indices = data_dim['category']['index']
                        dimension_indices[dim_id] = indices
                        continue
                
                # Fallback: Generate indices from metadata
                indices = {}
                for i, value in enumerate(dim_values.keys()):
                    indices[value] = i
                
                dimension_indices[dim_id] = indices
            
            return dimension_indices
            
        except Exception as e:
            logging.error(f"Error extracting dimension indices: {e}")
            return None
    
    def process_eurostat_json_improved(self, data: Dict, metadata: Dict, filters: Dict[str, str], 
                                       dimension_indices: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Improved process function for Eurostat JSON data to extract a time series based on filters
        
        Args:
            data: Eurostat API JSON response
            metadata: Dataset metadata
            filters: Filter dictionary
            dimension_indices: Mapping of dimension IDs to value-index pairs
            
        Returns:
            pd.DataFrame: Extracted time series
        """
        # Check if data contains values
        if "value" not in data or not data["value"]:
            logging.warning("No data values found in API response")
            return pd.DataFrame()
        
        # Get dimension order from data
        dimension_order = data.get("id", [])
        if not dimension_order:
            logging.error("No dimension order found in data")
            return pd.DataFrame()
        
        # Find time dimension
        time_dim = None
        time_index = -1
        for i, dim in enumerate(dimension_order):
            if dim.lower() in ('time_period', 'time'):
                time_dim = dim
                time_index = i
                break
        
        if time_dim is None:
            time_dim = dimension_order[-1]
            time_index = len(dimension_order) - 1
            logging.warning(f"No time dimension found in data, using last dimension: {time_dim}")
        
        # Create a mapping of time indices to time values
        time_values = {}
        if time_dim in dimension_indices:
            for time_code, idx in dimension_indices[time_dim].items():
                # Get time label from metadata
                time_label = None
                for dim in metadata.get('dimensions', []):
                    if dim.get('id') == time_dim and time_code in dim.get('values', {}):
                        time_label = dim['values'][time_code]
                        break
                
                if time_label is None:
                    time_label = time_code
                
                time_values[str(idx)] = time_label
        
        # Extract data matching the filters
        records = []
        
        # First, try to create expected position indices based on filters
        expected_position = [-1] * len(dimension_order)
        
        # Fill in expected positions for filter dimensions
        for dim_id, filter_value in filters.items():
            if dim_id not in dimension_indices:
                logging.warning(f"Dimension {dim_id} not found in dimension indices")
                continue
                
            indices = dimension_indices[dim_id]
            if filter_value not in indices:
                logging.warning(f"Value {filter_value} not found in indices for dimension {dim_id}")
                continue
                
            # Get the position of this dimension in the dimension order
            try:
                dim_position = dimension_order.index(dim_id)
                expected_position[dim_position] = indices[filter_value]
            except ValueError:
                logging.warning(f"Dimension {dim_id} not found in dimension order")
        
        # Log the expected position to help with debugging
        logging.debug(f"Expected position template: {expected_position}")
        
        # Process values based on the expected position
        for position_str, value in data["value"].items():
            # Parse position string
            if ':' in position_str:
                position = [int(p) for p in position_str.split(':')]
            else:
                position = [int(position_str)]
            
            # Check if position matches our filters
            # (excluding time dimension which we want all values for)
            matches = True
            
            # Ensure position has the correct length
            if len(position) != len(dimension_order):
                # Try to handle single position case
                if len(position) == 1 and len(dimension_order) == 1:
                    pass  # This is fine, single dimension
                else:
                    logging.debug(f"Position {position_str} has wrong length, expected {len(dimension_order)}")
                    continue
            
            # Check each dimension except time
            for i, pos in enumerate(position):
                if i == time_index:
                    continue  # Skip time dimension
                
                if expected_position[i] != -1 and pos != expected_position[i]:
                    matches = False
                    break
            
            if not matches:
                continue
            
            # If we're here, this position matches our filters
            # Extract time value and data value
            try:
                time_pos = position[time_index]
                time_label = time_values.get(str(time_pos), f"Period_{time_pos}")
                
                # Add to records
                records.append({
                    "period": time_label,
                    "value": value
                })
            except (IndexError, TypeError) as e:
                logging.warning(f"Error extracting time value from position {position_str}: {e}")
        
        # Convert to DataFrame
        if not records:
            logging.warning("No data found matching filters")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Set period as index and sort
        if "period" in df.columns:
            df = df.set_index("period")
            try:
                df = df.sort_index()
            except Exception as e:
                logging.warning(f"Error sorting by period: {e}")
        
        return df
    
    def export_time_series(self, df: pd.DataFrame, prefix: str = None, format: str = 'csv') -> Optional[str]:
        """Export a time series to a file"""
        if df is None or df.empty:
            logging.error("Cannot export empty DataFrame")
            return None
            
        dataset_id = df.attrs.get('dataset_id', 'unknown')
        
        if prefix:
            filename_prefix = f"{dataset_id}_{prefix}"
        else:
            filename_prefix = dataset_id
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            if format == 'csv':
                output_path = self.exports_dir / f"{filename_prefix}_{timestamp}.csv"
                df.to_csv(output_path)
            elif format == 'excel':
                output_path = self.exports_dir / f"{filename_prefix}_{timestamp}.xlsx"
                df.to_excel(output_path)
            elif format == 'json':
                output_path = self.exports_dir / f"{filename_prefix}_{timestamp}.json"
                df.reset_index().to_json(output_path, orient='records')
            else:
                logging.error(f"Unsupported export format: {format}")
                return None
                
            logging.info(f"Exported time series to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error exporting time series: {e}")
            return None
    
    def export_all_time_series(self, time_series_list: List[pd.DataFrame], format: str = 'csv') -> List[str]:
        """Export all time series to separate files"""
        exported_paths = []
        
        for i, df in enumerate(time_series_list):
            # Create a prefix including a hash of the filter values for uniqueness
            filters = df.attrs.get('filters', {})
            filter_hash = "_".join([f"{k}_{v}" for k, v in filters.items()])
            
            # Use a shorter prefix if the hash is too long
            if len(filter_hash) > 30:
                prefix = f"series_{i+1}"
            else:
                prefix = filter_hash
                
            path = self.export_time_series(df, prefix, format)
            if path:
                exported_paths.append(path)
        
        return exported_paths