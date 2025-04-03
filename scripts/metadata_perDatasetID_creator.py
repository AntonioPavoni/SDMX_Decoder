import requests
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
import time
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

class DatasetProcessor:
    def __init__(self, max_workers=10, rate_limit=0.8):
        self.base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"
        self.headers = {'Accept': 'application/xml'}
        self.ns = {
            's': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
            'c': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common',
            'gen': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic',
            'str': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/structurespecific'
        }
        self.output_dir = Path("C:/Users/Antonio/sdmx/eurostat_sdmx/data")
        self.datasets_dir = self.output_dir
        self.datasets_dir.mkdir(exist_ok=True, parents=True)
        self.setup_logging()
        self.load_reference_data()
        self.request_timeout = 60  # 60 second timeout for requests
        self.max_workers = max_workers
        self.rate_limit = rate_limit  # Minimum seconds between requests per thread

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataset_processor.log'),
                logging.StreamHandler(sys.stdout)  # Also log to console
            ]
        )

    def load_reference_data(self):
        """Load dataset list and codelist reference data"""
        try:
            with open(self.output_dir / 'dataset_list.json', 'r', encoding='utf-8') as f:
                self.dataset_catalog = json.load(f)
            with open(self.output_dir / 'codelist_all.json', 'r', encoding='utf-8') as f:
                self.codelist_catalog = json.load(f)['codelists']
            logging.info("Successfully loaded reference data")
        except Exception as e:
            logging.error(f"Error loading reference data: {e}")
            raise

    def find_dataset_index(self, dataset_id: str) -> int:
        """Find the index of a dataset in the catalog"""
        for i, dataset in enumerate(self.dataset_catalog['datasets']):
            if dataset['id'] == dataset_id:
                return i
        return -1  # Not found

    def get_dataset_structure(self, dataset_id: str) -> list:
        """Get the dimension structure for a dataset"""
        url = f"{self.base_url}/datastructure/ESTAT/{dataset_id}"
        try:
            time.sleep(random.uniform(0.1, self.rate_limit))  # Rate limiting with jitter
            response = requests.get(url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            dimensions = []
            dimension_list = root.find('.//s:DimensionList', self.ns)
            
            if dimension_list is not None:
                # Regular dimensions
                for dimension in dimension_list.findall('.//s:Dimension', self.ns):
                    dim_id = dimension.get('id')
                    position = int(dimension.get('position'))
                    
                    # Get codelist reference
                    codelist_ref = dimension.find('.//s:LocalRepresentation/s:Enumeration/Ref', self.ns)
                    codelist_id = codelist_ref.get('id') if codelist_ref is not None else None
                    
                    dimensions.append({
                        'id': dim_id,
                        'position': position,
                        'codelist_id': codelist_id
                    })
                
                # Time dimension
                time_dim = dimension_list.find('.//s:TimeDimension', self.ns)
                if time_dim is not None:
                    dimensions.append({
                        'id': 'TIME_PERIOD',
                        'position': int(time_dim.get('position')),
                        'codelist_id': None
                    })
            
            return sorted(dimensions, key=lambda x: x['position'])
        except Exception as e:
            logging.error(f"Error fetching structure for {dataset_id}: {e}")
            return []

    def get_dataset_values(self, dataset_id: str) -> Dict[str, Set[str]]:
        """Get actual values used in the dataset"""
        url = f"{self.base_url}/data/{dataset_id}"
        try:
            time.sleep(random.uniform(0.1, self.rate_limit))  # Rate limiting with jitter
            logging.info(f"Fetching data for {dataset_id}")
            response = requests.get(url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            
            # Check for empty response
            if not response.content.strip():
                logging.warning(f"Empty response for {dataset_id}")
                return {}
                
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                logging.error(f"XML parsing error for {dataset_id}: {e}")
                # Save the response content for debugging
                error_file = self.output_dir / f"{dataset_id}_error.xml"
                with open(error_file, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Saved error response to {error_file}")
                return {}
            
            dimension_values = defaultdict(set)
            time_values = set()
            
            # Generic format
            for series in root.findall('.//gen:Series', self.ns):
                series_key = series.find('.//gen:SeriesKey', self.ns)
                if series_key is not None:
                    for value in series_key.findall('.//gen:Value', self.ns):
                        concept = value.get('id')
                        val = value.get('value')
                        if concept and val:
                            dimension_values[concept].add(val)
                
                # Get time values
                for obs in series.findall('.//gen:Obs', self.ns):
                    time_value = None
                    time_elem = obs.find('.//gen:Time', self.ns)
                    if time_elem is not None and time_elem.text:
                        time_value = time_elem.text
                    if not time_value:
                        time_dim = obs.find('.//gen:ObsDimension', self.ns)
                        if time_dim is not None:
                            time_value = time_dim.get('value')
                    if not time_value:
                        time_value = obs.get('TIME_PERIOD')
                    
                    if time_value:
                        time_values.add(time_value)
            
            # Structure-specific format if needed
            if not dimension_values:
                for series in root.findall('.//str:Series', self.ns):
                    for key, value in series.attrib.items():
                        if key != 'VALUE_UPDATE' and key != 'TIME_PERIOD':
                            dimension_values[key].add(value)
                    
                    for obs in series.findall('.//str:Obs', self.ns):
                        time_value = obs.get('TIME_PERIOD')
                        if time_value:
                            time_values.add(time_value)
            
            if time_values:
                dimension_values['TIME_PERIOD'] = time_values
            
            return dimension_values
        except requests.exceptions.Timeout:
            logging.error(f"Request timeout for {dataset_id}")
            return {}
        except Exception as e:
            logging.error(f"Error fetching data for {dataset_id}: {e}")
            return {}

    def get_dataset_title(self, dataset_id: str) -> str:
        """Get dataset title from catalog"""
        for dataset in self.dataset_catalog['datasets']:
            if dataset['id'] == dataset_id:
                return dataset['title']
        return "Title not found"

    def create_dataset_metadata(self, dataset_id: str) -> Tuple[str, bool]:
        """Create metadata file for a single dataset - returns (dataset_id, success)"""
        try:
            # Check if file already exists
            output_file = self.datasets_dir / f"{dataset_id}_metadata.json"
            if output_file.exists():
                logging.info(f"Metadata file for {dataset_id} already exists, skipping")
                return dataset_id, True
                
            # Get dataset structure (dimensions and their codelists)
            structure = self.get_dataset_structure(dataset_id)
            if not structure:
                logging.error(f"No structure found for {dataset_id}")
                return dataset_id, False
            
            # Get actual values used in the dataset
            actual_values = self.get_dataset_values(dataset_id)
            if not actual_values:
                logging.error(f"No values found for {dataset_id}")
                return dataset_id, False
            
            # Create metadata
            metadata = {
                'id': dataset_id,
                'title': self.get_dataset_title(dataset_id),
                'dimensions': []
            }
            
            # Process each dimension
            for dim in structure:
                dim_id = dim['id']
                codelist_id = dim['codelist_id']
                used_values = actual_values.get(dim_id, set())
                
                # Get descriptions for values
                values_with_desc = {}
                if dim_id == 'TIME_PERIOD':
                    # Handle time periods
                    values_with_desc = {value: value for value in used_values}
                elif codelist_id and codelist_id in self.codelist_catalog:
                    # Map values to descriptions from codelist
                    codelist = self.codelist_catalog[codelist_id]
                    values_with_desc = {
                        value: codelist.get(value, value)
                        for value in used_values
                    }
                else:
                    # Fallback if no codelist found
                    values_with_desc = {value: value for value in used_values}
                
                if values_with_desc:
                    metadata['dimensions'].append({
                        'id': dim_id,
                        'position': dim['position'],
                        'values': values_with_desc
                    })
            
            # Save metadata file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Created metadata file for {dataset_id}")
            return dataset_id, True
            
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_id}: {e}")
            return dataset_id, False

    def save_progress(self, processed_datasets: List[str]):
        """Save the current progress to a file"""
        progress_file = self.output_dir / 'processing_progress.json'
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({'processed_datasets': processed_datasets}, f)
        logging.info(f"Saved progress with {len(processed_datasets)} processed datasets")

    def load_progress(self) -> List[str]:
        """Load the processed datasets list"""
        progress_file = self.output_dir / 'processing_progress.json'
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    return progress.get('processed_datasets', [])
            except Exception as e:
                logging.error(f"Error loading progress: {e}")
        return []

    def process_datasets_parallel(self, start_index: int = 0, num_datasets: int = None, 
                                save_interval: int = 20):
        """Process multiple datasets in parallel"""
        # Get list of datasets
        datasets = self.dataset_catalog['datasets']
        
        # Determine range to process
        if num_datasets is None:
            end_index = len(datasets)
        else:
            end_index = min(start_index + num_datasets, len(datasets))
        
        datasets_to_process = datasets[start_index:end_index]
        total = len(datasets_to_process)
        
        # Load already processed datasets
        processed_datasets = set(self.load_progress())
        skipped_count = 0
        
        # Filter out already processed datasets
        to_process = []
        for dataset in datasets_to_process:
            dataset_id = dataset['id']
            if dataset_id in processed_datasets or (self.datasets_dir / f"{dataset_id}_metadata.json").exists():
                skipped_count += 1
            else:
                to_process.append(dataset_id)
        
        logging.info(f"Found {len(to_process)} datasets to process, skipping {skipped_count} already processed")
        
        # Process in parallel
        results = []
        batch_size = save_interval
        batches = [to_process[i:i+batch_size] for i in range(0, len(to_process), batch_size)]
        
        with tqdm(total=len(to_process), desc="Processing datasets") as progress_bar:
            for batch_idx, batch in enumerate(batches):
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.create_dataset_metadata, dataset_id): dataset_id 
                              for dataset_id in batch}
                    
                    for future in as_completed(futures):
                        dataset_id, success = future.result()
                        if success:
                            processed_datasets.add(dataset_id)
                        results.append((dataset_id, success))
                        progress_bar.update(1)
                
                # Save progress after each batch
                self.save_progress(list(processed_datasets))
                
                # Small delay between batches to be even more respectful to the API
                if batch_idx < len(batches) - 1:  # If not the last batch
                    time.sleep(1)
        
        # Final stats
        success_count = sum(1 for _, success in results if success)
        logging.info(f"Completed processing {len(results)} datasets, {success_count} successful, {len(results) - success_count} failed")
        return results

def main():
    parser = argparse.ArgumentParser(description='Process Eurostat datasets to create metadata files')
    parser.add_argument('--start-index', type=int, help='Start processing from this index')
    parser.add_argument('--start-dataset', type=str, help='Start processing from this dataset ID')
    parser.add_argument('--num-datasets', type=int, help='Number of datasets to process')
    parser.add_argument('--save-interval', type=int, default=20, help='Save progress every N datasets')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--rate-limit', type=float, default=1.0, 
                       help='Minimum seconds between requests per thread (default: 1.0)')
    args = parser.parse_args()
    
    processor = DatasetProcessor(max_workers=args.workers, rate_limit=args.rate_limit)
    
    start_index = args.start_index or 0
    if args.start_dataset:
        dataset_index = processor.find_dataset_index(args.start_dataset)
        if dataset_index >= 0:
            start_index = dataset_index
            print(f"Starting from dataset {args.start_dataset} at index {start_index}")
        else:
            print(f"Dataset {args.start_dataset} not found, starting from index {start_index}")
    
    try:
        processor.process_datasets_parallel(
            start_index=start_index,
            num_datasets=args.num_datasets,
            save_interval=args.save_interval
        )
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        logging.info("Progress has been saved. You can resume later.")

if __name__ == "__main__":
    main()