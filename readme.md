# Eurostat SDMX Time Series Extractor 📊

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 

Streamlined access to Eurostat time series data via SDMX, providing ready-to-use time series datasets directly. This project aims to simplify the process of fetching and decoding SDMX data, offering an alternative focused on direct time series generation.

## Motivation 

Working with SDMX data, especially from large providers like Eurostat, can involve complex steps of fetching data, understanding metadata (codelists, data structure definitions), and decoding observations into meaningful time series. While tools like `pandasdmx` exist, they don't work, this project focuses on leveraging pre-compiled metadata and efficient caching to provide users with direct access to fully decoded time series data for specific Eurostat datasets.

The goal is to abstract away the complexities of SDMX structure and provide a simpler interface for obtaining analysis-ready time series. 

## Features

*   **Direct Time Series Generation:** Obtain Pandas DataFrames representing time series directly from Eurostat dataset IDs.
*   **Metadata-Driven:** Utilizes pre-compiled metadata (dataset structures, codelists) for accurate data decoding.
*   **Efficient Caching:** Caches downloaded SDMX data to speed up subsequent requests.
*   **Eurostat Focused:** Currently tailored for accessing data from the Eurostat SDMX API.
*   **Extensible:** Designed with the potential to support other SDMX providers in the future.

##  Installation

Currently, the project can be used by cloning the repository:

```bash
git clone https://github.com/YOUR_USERNAME/EUROSTAT_SDMX.git # Replace with your repo URL
cd EUROSTAT_SDMX
# Install dependencies (Add a requirements.txt or setup.py)
# pip install -r requirements.txt
```

*(TODO: Add package installation instructions if you publish it to PyPI)*

## 💡 Usage

*(Provide clear examples here once the core API is stable)*

**Example 1: Fetching and getting time series for a dataset**

```python
# Example pseudocode - adjust based on your actual implementation
from src.DatasetIDAllData_Extraction_toCache import fetch_data_for_id
from src.timeSeriesExtraction import generate_time_series

dataset_id = "nama_10_gdp" # Example Eurostat dataset ID

# Fetch and cache data (if not already cached)
fetch_data_for_id(dataset_id)

# Generate time series using cached data and metadata
time_series_df = generate_time_series(dataset_id)

print(time_series_df.head())
```

**Example 2: Exploring available datasets**

*(Show how a user might list available datasets based on `dataset_list.json`)*

```python
# Example pseudocode
import json

with open("data/dataset_list.json", 'r') as f:
    datasets = json.load(f)

print(f"Available datasets: {len(datasets)}")
# Potentially show a few dataset titles/IDs
```

##  Project Structure

```
EUROSTAT_SDMX/
├── data/
│   ├── cache/                  # Cached SDMX data files
│   ├── codelists/              # Compiled codelists (consider consolidating or explaining purpose)
│   ├── datasets/               # Pre-processed metadata per dataset ID
│   ├── codelist_all.json       # Combined codelist (if used)
│   └── dataset_list.json       # List of discoverable Eurostat dataset IDs & titles
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── scripts/                    # Utility scripts for metadata generation, etc.
│   ├── codelist_catalog_creator.py
│   ├── dataset_catalog_creator.py
│   └── metadata_perDatasetID_creator.py
├── src/                        # Core source code
│   ├── DatasetIDAllData_Extraction_toCache.py # Data fetching and caching logic
│   └── timeSeriesExtraction.py # Time series generation logic
├── tests/                      # Unit and integration tests
│   └── test_extraction.py
├── .gitignore                  # Git ignore file
├── LICENSE                     # Project license file (e.g., MIT)
├── README.md                   # This file
└── requirements.txt            # Python dependencies (Recommended)
└── processing_progress.json    # Tracks processing state (optional detail)

```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.