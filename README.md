# Unmasking and Exploiting Multi-Encryption Weaknesses in Distributed Data Protection Middleware

## Project Overview

This project conducts in-depth research on multi-encryption scheme vulnerabilities in distributed data protection middleware. It proposes and implements various attack methods and defense measures, enabling dynamic ciphertext header inference without prior knowledge. Through experiments on real-world datasets, the project validates the vulnerabilities of existing encryption schemes in attribute recovery, frequency analysis, and other aspects.

## Key Features

- **Dynamic Column Mapping**: Attack schemes can now infer ciphertext column mappings without knowing the ciphertext headers in advance, by using `AttributeRecoverAttack` combined with `matrix_plain`
- **Multiple Attack Vectors**: Supports OPE, DET, SSE, and cumulative attacks
- **Defense Mechanisms**: Provides obfuscation and padding countermeasures
- **Multi-Dataset Validation**: Experiments on PUDF, ACS, Alzheimer, and USA Crime datasets
- **Complete Dataset Pipeline**: Includes dataset preprocessing tools for data cleaning, formatting, merging, and analysis

## Project Structure

```
CCS2026/
├── NKW15andSSESchemes/      # Existing attack scheme implementations
│   ├── AttributeRecoverAttack.py      # Attribute recovery attack
│   ├── FrequencyAnalysisAttack.py     # Frequency analysis attack
│   ├── CumulativeAttack.py           # Cumulative attack
│   ├── Oya2021.py                  # Oya et al. 2021 attack
│   ├── Oya2021_A1.py               # Oya et al. 2021 auxiliary data attack
│   ├── Oya2021_SSE.py              # Oya et al. 2021 SSE attack
│   ├── NWX24_A1.py                 # NWX24 auxiliary data attack
│   ├── NWX24_SSE.py                # NWX24 SSE attack
│   └── KNW_jigsaw.py               # Jigsaw attack
├── Ours/                        # Proposed improved attack schemes
│   ├── AttributeRecoveryAttack.py          # Improved attribute recovery attack
│   ├── EnhancedFrequencyAnalysisAttack.py  # Enhanced frequency analysis attack
│   ├── EnhancedCumulativeAttack.py         # Enhanced cumulative attack
│   ├── AttackUnderAssumption1.py          # Attack under assumption 1
│   ├── AttackUnderAssumption2.py          # Attack under assumption 2
│   └── AttackusingAuxiliary.py            # Attack using auxiliary data
├── Countermeasures/            # Defense measures
│   ├── Obfuscation.py               # Obfuscation defense
│   ├── padding.py                  # Padding defense
│   ├── Obfuscation_A1.py           # Obfuscation defense (auxiliary data)
│   ├── padding_A1.py               # Padding defense (auxiliary data)
│   ├── AttackUsingAuxiliaryObfuscation.py  # Auxiliary data-based obfuscation attack
│   ├── AttackUsingAuxiliaryPadding.py      # Auxiliary data-based padding attack
│   ├── PotenialAttackUsingAuxiliary.py     # Potential auxiliary data attack
│   └── potenialCountermeatures.py         # Potential countermeasures
├── otherdataset/               # Experiments on different datasets
│   ├── otherDataset_ours/            # Experiments using our scheme
│   │   ├── 2018.py                  # PUDF 2018 dataset experiment
│   │   ├── ACS.py                   # ACS dataset experiment
│   │   └── Alzheimer.py             # Alzheimer dataset experiment
│   ├── otherDataset_jigsaw/          # Experiments using Jigsaw scheme
│   │   ├── 2018.py
│   │   ├── ACS.py
│   │   └── Alzheimer.py
│   └── otherDataset_OK21/           # Experiments using Oya 2021 scheme
│       ├── 2018.py
│       ├── ACS.py
│       └── Alzheimer.py
├── dataset/                     # Dataset files
│   ├── 2015.csv                         # PUDF 2015 plaintext data
│   ├── PUDF_base1_4q2018.csv           # PUDF 2018 ciphertext data
│   ├── PUDF_base1_4q2019.csv           # PUDF 2019 ciphertext data
│   ├── Alzheimer_plain.csv              # Alzheimer plaintext data
│   ├── Alzheimer_cipher.csv             # Alzheimer ciphertext data
│   ├── usa_2_plain.csv                # USA crime data plaintext
│   └── usa_2_cipher.csv               # USA crime data ciphertext
├── Dataset solve/                # Dataset preprocessing utilities
│   ├── 1 DataSolve.py               # Convert TXT to CSV, ICD code to disease names
│   ├── 2 deletena.py               # Remove NA values
│   ├── 3 xiugaiAge.py              # Age data formatting
│   ├── 4 xugaiAge.py              # Age data correction
│   ├── 5 mergeCSV.py              # Merge multiple CSV files
│   ├── DataAnalysis.py           # Dataset statistical analysis
│   ├── dataset_extraction.ipynb  # Dataset extraction notebook
│   ├── dataset_solve.ipynb       # Dataset processing notebook
│   ├── new_dataset.py            # Generate new dataset format
│   └── diagnosis_codes_ICD_10_CM.json  # ICD-10 diagnosis codes reference
├── frequency/                    # Frequency data files (hospital/query statistics)
│   └── *.csv                    # Individual frequency files for each entity
├── functions.py                # Core utility function library
├── README.md                   # Project documentation
└── requirements.txt           # Python dependencies
```

## Core Features

### 1. Attack Schemes

#### 1.1 Attribute Recovery Attack
- **Files**: `NKW15andSSESchemes/AttributeRecoverAttack.py`, `Ours/AttributeRecoveryAttack.py`
- **Functionality**: Infers the mapping between ciphertext columns and plaintext columns by analyzing features such as column element frequency and unique value counts
- **Features**:
  - Supports dynamic column mapping inference without prior knowledge of ciphertext headers
  - Based on Euclidean distance and feature vector similarity matching
  - Applicable to OPE, DET, SSE encryption schemes

#### 1.2 Frequency Analysis Attack
- **Files**: `NKW15andSSESchemes/FrequencyAnalysisAttack.py`, `Ours/EnhancedFrequencyAnalysisAttack.py`
- **Functionality**: Performs DET attacks using column element frequency distributions with ℓ₂-optimization
- **Features**:
  - Supports plaintext/ciphertext spaces of different sizes through virtual element padding
  - Uses Hungarian algorithm for optimal mapping matching
  - Enhanced version supports auxiliary data attacks

#### 1.3 Cumulative Attack
- **Files**: `NKW15andSSESchemes/CumulativeAttack.py`, `Ours/EnhancedCumulativeAttack.py`
- **Functionality**: Gradually infers more plaintext by accumulating recovered information
- **Features**:
  - Based on co-occurrence frequency and graph algorithms
  - Supports weighted parameters (e.g., [0.5, 0.1, 0.4])
  - Enhanced version introduces auxiliary data

#### 1.4 SSE Attack
- **Files**: `NKW15andSSESchemes/Oya2021_SSE.py`, `NKW15andSSESchemes/NWX24_SSE.py`
- **Functionality**: Attacks against searchable encryption schemes
- **Features**:
  - Utilizes query frequency and document frequency
  - Based on cost matrix and Hungarian algorithm

### 2. Defense Measures

#### 2.1 Obfuscation
- **Files**: `Countermeasures/Obfuscation.py`, `Countermeasures/Obfuscation_A1.py`
- **Functionality**: Obscures real data by adding fake records
- **Features**:
  - Supports different obfuscation ratios
  - Maintains data statistical properties

#### 3.2 Padding
- **Files**: `Countermeasures/padding.py`, `Countermeasures/padding_A1.py`
- **Functionality**: Increases data volume by padding with fake records
- **Features**:
  - Dynamically adjusts padding ratio
  - Maintains data distribution characteristics

#### 2.3 Vertical Partitioning
- **Files**: `Countermeasures/potenialCountermeasures.py`, `Countermeasures/potenialCountermeasures_A1.py`
- **Functionality**: Increases data volume by padding with fake records
- **Features**:
  - Dynamically adjusts padding ratio
  - Maintains data distribution characteristics

## Dataset Preprocessing (Dataset solve)

The `Dataset solve/` directory contains comprehensive tools for data cleaning, formatting, and preparation:

### Available Scripts

| File | Description |
|------|-------------|
| `1 DataSolve.py` | Converts TXT files to CSV format and maps ICD diagnosis codes to human-readable disease names |
| `2 deletena.py` | Removes NA/NULL values from datasets |
| `3 xiugaiAge.py` | Formats age data into standardized categories (e.g., "0 year", "5 year", "10 year") |
| `4 xugaiAge.py` | Corrects and validates age data entries |
| `5 mergeCSV.py` | Merges multiple CSV files into a single combined dataset |
| `DataAnalysis.py` | Performs statistical analysis on datasets |
| `new_dataset.py` | Generates new dataset formats for specific use cases |
| `dataset_extraction.ipynb` | Jupyter notebook for dataset extraction workflows |
| `dataset_solve.ipynb` | Jupyter notebook for comprehensive dataset processing |
| `diagnosis_codes_ICD_10_CM.json` | Reference file containing ICD-10-CM diagnosis codes and descriptions |

### Data Processing Pipeline

```
Raw Data (TXT/JSON)
    ↓
1 DataSolve.py (TXT → CSV, ICD codes → Disease names)
    ↓
2 deletena.py (Remove NA values)
    ↓
3 xiugaiAge.py / 4 xugaiAge.py (Age formatting)
    ↓
5 mergeCSV.py (Merge datasets)
    ↓
DataAnalysis.py (Statistical analysis)
    ↓
Processed Dataset (CSV) → Ready for encryption/attacks
```

## Dataset Description

### PUDF (Patient Utilization Data Files)
- **Plaintext**: `dataset/2015.csv`
- **Ciphertext**: `dataset/PUDF_base1_4q2018.csv`, `dataset/PUDF_base1_4q2019.csv`
- **Download**: [NYC Health + Hospitals Patient Utilization Data](https://www1.nyc.gov/site/hhc/data-transparency.page)
- **Main Fields**: Age, Gender, Risk, Admission Type, Race, Hospital, Principal Diagnosis
- **Encryption Types**: OPE, DET, SSE

### ACS (American Community Survey)
- **Plaintext/Ciphertext**: Requires preparation (see download link)
- **Download**: [U.S. Census Bureau ACS](https://www.census.gov/programs-surveys/acs/data)
- **Main Fields**: VALUEH, AGE, SEX, MARST, RACE, EDUC, OCC, IND
- **Encryption Types**: OPE, DET, SSE

### Alzheimer
- **Plaintext**: `dataset/Alzheimer_plain.csv`
- **Ciphertext**: `dataset/Alzheimer_cipher.csv`
- **Download**: [CDC Alzheimer's Disease Data](https://www.cdc.gov/aging/agingdata/alzheimers-data-portal.html)
- **Main Fields**: YearStart, LocationAbbr, Stratification2, Class, DataValueTypeID, Topic
- **Encryption Types**: OPE, DET, SSE

### USA Crime Data
- **Plaintext**: `dataset/usa_2_plain.csv`
- **Ciphertext**: `dataset/usa_2_cipher.csv`
- **Download**: [Los Angeles Crime Data](https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z)
- **Encryption Types**: OPE, DET

## Experimental Setup

### Experiment Settings
- Each scale runs 50 experiments
- Calculate average recovered keywords and success rate

### Evaluation Metrics
- **Keyword Count**: Total unique values in the dataset
- **Successfully Recovered Keywords**: Number of plaintext values recovered
- **Recovery Rate**: Recovered / Total keywords

## Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Dataset Preprocessing
```bash
# Convert and format PUDF data
python "Dataset solve/1 DataSolve.py"

# Remove NA values
python "Dataset solve/2 deletena.py"

# Format age data
python "Dataset solve/3 xiugaiAge.py"
python "Dataset solve/4 xugaiAge.py"

# Merge CSV files
python "Dataset solve/5 mergeCSV.py"

# Analyze dataset statistics
python "Dataset solve/DataAnalysis.py"
```

### 3. Running Attack Experiments

#### Using Our Scheme (Dynamic Header Inference)
```bash
# PUDF 2010 dataset
python Ours/AttackUnderAssumption1.py # Assumption 1
python Ours/AttackUnderAssumption2.py # Assumption 2

# PUDF 2018, 2019 dataset
python otherdataset/otherDataset_ours/2018.py

# ACS dataset
python otherdataset/otherDataset_ours/ACS.py

# Alzheimer dataset
python otherdataset/otherDataset_ours/Alzheimer.py
```

#### Using NWX24 Scheme
```bash
# PUDF 2010, 2018, 2019 dataset
python otherdataset/otherDataset_jigsaw/2018.py

# ACS dataset
python otherdataset/otherDataset_jigsaw/ACS.py

# Alzheimer dataset
python otherdataset/otherDataset_jigsaw/Alzheimer.py
```

#### Using OK21 Scheme

```bash
# PUDF 2010, 2018, 2019 dataset
python otherdataset/otherDataset_OK21/2018.py

# ACS dataset
python otherdataset/otherDataset_OK21/ACS.py

# Alzheimer dataset
python otherdataset/otherDataset_OK21/Alzheimer.py
```

### 4. Running Defense Experiments
```bash
# Obfuscation defense
python Countermeasures/Obfuscation.py

# Padding defense
python Countermeasures/padding.py
```

## Core Functions

### functions.py Utility Library

#### Data Processing
- `read_csv_to_matrix(file_path)` - Read CSV to matrix
- `generate_submatrix(matrix, columns)` - Generate submatrix by columns
- `random_extract(matrix, num_rows)` - Random row extraction
- `replace_nested_inplace(matrix, old_value, new_value)` - In-place replacement

#### Feature Computation
- `compute_feature_vector(matrix, select_column, dataset_name)` - Feature vectors
- `compute_statistic(data)` - Frequency and CDF statistics
- `count_frequency_multicol(matrix, column_names)` - Multi-column frequencies

#### Mapping & Attack
- `find_optimal_mapping(data_c, data_z)` - Optimal mapping via Hungarian algorithm
- `find_closest_mapping(dict1, dict2)` - Closest mapping by distance
- `l2_optimization_attack(plaintext_hist, ciphertext_hist)` - ℓ₂-optimization attack

#### Helpers
- `create_rowid_dict(matrix, header, record_id_column)` - Row ID mapping
- `find_unique_value(input_list)` - Unique value detection
- `count_keywords(matrix)` - Keyword counting

## Key Contributions

1. **Dynamic Attribute Recovery**: Attack schemes can now infer column mappings without prior ciphertext header knowledge

2. **Enhanced Attack Algorithms**: Improved frequency analysis with ℓ₂-optimization and virtual element padding

3. **Multi-Dataset Validation**: Experiments on PUDF, ACS, Alzheimer, and USA Crime datasets

4. **Defense Measures**: Obfuscation and padding countermeasures with effectiveness evaluation

5. **Auxiliary Data Attacks**: Investigation of attacks when auxiliary data is available

6. **Complete Data Pipeline**: Comprehensive dataset preprocessing tools for real-world data preparation

## Experimental Results

Results demonstrate that:
1. Attribute recovery attacks effectively infer ciphertext-plaintext column mappings
2. Enhanced attacks outperform existing schemes across multiple datasets
3. Obfuscation and padding reduce attack success rates
4. Auxiliary data availability significantly impacts attack effectiveness

## License

This project is for academic research use only.

## Acknowledgments

This work was supported by [Funding Agency/Grant Number]. Dataset providers: NYC Health + Hospitals, U.S. Census Bureau, CDC, Los Angeles Open Data Portal.
