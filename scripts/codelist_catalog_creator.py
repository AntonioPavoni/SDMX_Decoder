import requests
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
import time

class CodelistCatalogCreator:
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
            filename='codelist_catalog.log'
        )

    def get_codelists(self):
        """Get all codelists directly"""
        url = f"{self.base_url}/codelist/ESTAT"
        try:
            logging.info("Fetching codelist catalog from Eurostat")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            catalog = {
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'codelists': {}
            }
            
            for codelist in root.findall('.//s:Codelist', self.ns):
                codelist_id = codelist.get('id')
                if not codelist_id:
                    continue
                
                values = {}
                for code in codelist.findall('.//s:Code', self.ns):
                    code_id = code.get('id')
                    for name in code.findall('.//c:Name', self.ns):
                        if name.get('{http://www.w3.org/XML/1998/namespace}lang', '').lower() == 'en':
                            values[code_id] = name.text
                            break
                
                if values:
                    catalog['codelists'][codelist_id] = values
                    logging.info(f"Processed codelist: {codelist_id}")
            
            catalog['total_codelists'] = len(catalog['codelists'])
            
            # Save catalog
            output_file = self.output_dir / 'codelist_all.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved {len(catalog['codelists'])} codelists to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating codelist catalog: {e}")
            return False

if __name__ == "__main__":
    creator = CodelistCatalogCreator()
    creator.get_codelists()