import os
import json
import requests
import logging
import gzip
import time
from pathlib import Path
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Ensure the script can find modules in the project
current_dir = Path(__file__).parent
root_dir = current_dir.parent  # This would be the EUROSTAT_SDMX directory
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

class DatasetExtractor:
    def __init__(self, base_dir: str = None, max_retries: int = 3, retry_delay: int = 5):
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
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"
        self.headers = {'Accept': 'application/json'}
        
        # Add retry mechanism parameters
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Add checkpoint file path for resuming operations
        self.checkpoint_file = self.base_dir / "checkpoint.json"
        
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_path = root_dir / "src" / "dataset_extraction.log"
        
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
    
    def load_dataset_metadata(self, dataset_id: str):
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
    
    def fetch_and_cache_dataset(self, dataset_id: str, force_refresh: bool = False):
        """Fetch a dataset from Eurostat API and cache it"""
        cache_path = self.cache_dir / f"{dataset_id}_data.json.gz"
        
        # Check if we already have cached data and force_refresh is not set
        if cache_path.exists() and not force_refresh:
            logging.info(f"Dataset {dataset_id} already cached at {cache_path}")
            try:
                # Verify the cached file is valid
                with gzip.open(cache_path, 'rb') as f:
                    # Just read a small portion to check the file is valid
                    f.read(100)
                return True
            except Exception as e:
                logging.warning(f"Cached file for {dataset_id} appears corrupted: {e}. Will re-fetch.")
        
        # Load metadata to get basic info about the dataset
        metadata = self.load_dataset_metadata(dataset_id)
        if not metadata:
            logging.error(f"Cannot proceed with {dataset_id}: metadata not found.")
            return False
        
        # Fetch data from Eurostat API
        logging.info(f"Fetching data for {dataset_id} from Eurostat API...")
        
        url = f"{self.base_url}/data/{dataset_id}?format=json"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=300)
                response.raise_for_status()
                
                data = response.json()
                
                # Add metadata info to the cached data for easier reference
                data['_metadata'] = {
                    'dataset_id': dataset_id,
                    'title': metadata.get('title', 'Unknown'),
                    'cached_date': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save to gzipped JSON cache
                logging.info(f"Saving data for {dataset_id} to cache...")
                with gzip.open(cache_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f)
                
                logging.info(f"Successfully cached data for {dataset_id} at {cache_path}")
                return True
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (attempt + 1)
                    logging.warning(f"Attempt {attempt+1} failed for {dataset_id}. Retrying in {delay} seconds... Error: {e}")
                    time.sleep(delay)
                else:
                    logging.error(f"Failed to fetch data for {dataset_id} after {self.max_retries} attempts: {e}")
                    return False
            except Exception as e:
                logging.error(f"Unexpected error processing {dataset_id}: {e}")
                return False
    
    def save_checkpoint(self, current_dataset_id, dataset_ids):
        """Save checkpoint information to resume later"""
        checkpoint_data = {
            'last_processed_id': current_dataset_id,
            'last_processed_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'remaining_datasets': dataset_ids
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            logging.info(f"Checkpoint saved: last processed dataset was {current_dataset_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self):
        """Load checkpoint information to resume processing"""
        if not self.checkpoint_file.exists():
            logging.info("No checkpoint file found. Starting from beginning.")
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logging.info(f"Loaded checkpoint: last processed dataset was {checkpoint_data.get('last_processed_id')}")
            return checkpoint_data
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None
    
    def process_multiple_datasets(self, dataset_ids, force_refresh=False, max_workers=4, start_from=None):
        """Process multiple datasets in parallel, with option to resume from a specific dataset ID"""
        results = {}
        
        # If start_from is specified, find its index in the dataset_ids list
        start_index = 0
        if start_from:
            try:
                start_index = dataset_ids.index(start_from)
                dataset_ids = dataset_ids[start_index:]
                logging.info(f"Resuming from dataset {start_from} (skipping {start_index} datasets)")
            except ValueError:
                logging.warning(f"Dataset ID {start_from} not found in the list. Starting from the beginning.")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.fetch_and_cache_dataset, dataset_id, force_refresh): dataset_id 
                for dataset_id in dataset_ids
            }
            
            with tqdm(total=len(dataset_ids), desc="Processing datasets") as progress:
                for future in as_completed(future_to_dataset):
                    dataset_id = future_to_dataset[future]
                    try:
                        results[dataset_id] = future.result()
                        # Save checkpoint after each dataset
                        remaining = [d for d in dataset_ids if d not in results]
                        self.save_checkpoint(dataset_id, remaining)
                    except Exception as e:
                        logging.error(f"Exception processing dataset {dataset_id}: {e}")
                        results[dataset_id] = False
                        # Save checkpoint even on failure
                        remaining = [d for d in dataset_ids if d not in results]
                        self.save_checkpoint(dataset_id, remaining)
                    
                    progress.update(1)
        
        return results
    
    def load_dataset_list(self, list_file="dataset_list.json"):
        """Load the list of dataset IDs from dataset_list.json"""
        # Try in data directory first
        list_path = self.base_dir / list_file
        if not list_path.exists():
            # Try in datasets subdirectory as fallback
            list_path = self.datasets_dir / list_file
            if not list_path.exists():
                logging.error(f"Dataset list file not found at {self.base_dir / list_file} or {self.datasets_dir / list_file}")
                return []
        
        try:
            with open(list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle the specific structure in your JSON file
            if isinstance(data, dict) and 'datasets' in data and isinstance(data['datasets'], list):
                dataset_list = data['datasets']
                if all(isinstance(item, dict) and 'id' in item for item in dataset_list):
                    return [item['id'] for item in dataset_list]
            
            # Handle other formats as before
            elif isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data
                elif all(isinstance(item, dict) and 'id' in item for item in data):
                    return [item['id'] for item in data]
                elif all(isinstance(item, dict) and 'datasetID' in item for item in data):
                    return [item['datasetID'] for item in data]
            
            logging.error(f"Invalid format in dataset list file: {list_path}")
            return []
            
        except Exception as e:
            logging.error(f"Error loading dataset list: {e}")
            return []
        
        
def main():
    parser = argparse.ArgumentParser(description='Fetch and cache Eurostat datasets')
    parser.add_argument('--dataset', type=str, help='Single dataset ID to process')
    parser.add_argument('--subset', type=int, default=0, help='Process a subset of N datasets from the list file')
    parser.add_argument('--list-file', type=str, default='dataset_list.json', help='JSON file containing dataset IDs')
    parser.add_argument('--data-dir', type=str, help='Base data directory path')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh data from API even if cached')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads for parallel processing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--start-from', type=str, help='Resume from a specific dataset ID')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = DatasetExtractor(base_dir=args.data_dir)
    
    # Process a single dataset if specified
    if args.dataset:
        print(f"\nProcessing dataset: {args.dataset}")
        success = extractor.fetch_and_cache_dataset(args.dataset, args.force_refresh)
        
        if success:
            print(f"Successfully fetched and cached dataset {args.dataset}")
        else:
            print(f"Failed to process dataset {args.dataset}")
        
        return
    
    # Process multiple datasets from the list file
    dataset_ids = extractor.load_dataset_list(args.list_file)
    if not dataset_ids:
        logging.error(f"No dataset IDs found in list file")
        return
        
    print(f"Found {len(dataset_ids)} datasets in list file")
    
    # Process a subset if specified
    if args.subset > 0:
        dataset_ids = dataset_ids[:min(args.subset, len(dataset_ids))]
        print(f"Processing subset of {len(dataset_ids)} datasets")
    
    # Handle resuming from checkpoint or specific dataset ID
    start_from = None
    if args.resume:
        checkpoint = extractor.load_checkpoint()
        if checkpoint and 'remaining_datasets' in checkpoint and checkpoint['remaining_datasets']:
            # Use the remaining datasets from the checkpoint
            dataset_ids = checkpoint['remaining_datasets']
            print(f"Resuming with {len(dataset_ids)} datasets from checkpoint")
        elif checkpoint and 'last_processed_id' in checkpoint:
            # If we have a last processed ID but no remaining list, find it in the full list
            last_id = checkpoint['last_processed_id']
            try:
                last_index = dataset_ids.index(last_id)
                dataset_ids = dataset_ids[last_index + 1:]  # Start from the next one
                print(f"Resuming from dataset after {last_id} (skipping {last_index + 1} datasets)")
            except ValueError:
                print(f"Last processed dataset {last_id} not found in current list. Starting from beginning.")
    elif args.start_from:
        start_from = args.start_from
        print(f"Will attempt to start processing from dataset ID: {start_from}")
    
    results = extractor.process_multiple_datasets(dataset_ids, args.force_refresh, args.workers, start_from)
    
    # Clean up checkpoint file after successful completion
    if extractor.checkpoint_file.exists():
        try:
            extractor.checkpoint_file.unlink()
            print("Processing completed successfully. Checkpoint file removed.")
        except Exception as e:
            logging.warning(f"Could not remove checkpoint file: {e}")
    
    # Summarize results
    success_count = sum(1 for success in results.values() if success)
    print(f"\nSuccessfully processed {success_count} out of {len(results)} datasets")
    
    # List failed datasets if any
    failed = [ds_id for ds_id, success in results.items() if not success]
    if failed:
        print(f"Failed to process {len(failed)} datasets:")
        for ds_id in failed[:10]:  # Show first 10
            print(f"  - {ds_id}")
        if len(failed) > 10:
            print(f"  - ... and {len(failed) - 10} more")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        print("You can resume later using the --resume flag")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print("\nScript encountered an error. You can resume later using the --resume flag")