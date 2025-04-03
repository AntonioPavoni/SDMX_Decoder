import requests
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
import time

class DatasetCatalogCreator:
    def __init__(self):
        self.base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"
        self.headers = {'Accept': 'application/xml'}
        self.ns = {
            's': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
            'c': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
        }
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='dataset_catalog.log'
        )

    def create_catalog(self):
        """Create the complete dataset catalog"""
        url = f"{self.base_url}/dataflow/ESTAT"
        try:
            logging.info("Fetching dataset catalog from Eurostat")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            datasets = []
            for dataflow in root.findall('.//s:Dataflow', self.ns):
                dataset_id = dataflow.get('id')
                title = None
                
                # Get English title
                for name in dataflow.findall('.//c:Name', self.ns):
                    if name.get('{http://www.w3.org/XML/1998/namespace}lang', '').lower() == 'en':
                        title = name.text
                        break
                
                if dataset_id and title:
                    datasets.append({
                        'id': dataset_id,
                        'title': title
                    })
            
            # Sort by dataset ID
            datasets.sort(key=lambda x: x['id'])
            
            # Save to file
            output_file = self.output_dir / 'dataset_list.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_datasets': len(datasets),
                    'datasets': datasets
                }, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved {len(datasets)} datasets to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating catalog: {e}")
            return False

if __name__ == "__main__":
    creator = DatasetCatalogCreator()
    creator.create_catalog()