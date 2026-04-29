# Unmasking and Exploiting Multi-Encryption Weaknesses in Distributed Data Protection Middleware

## Project Overview

This project conducts in-depth research on multi-encryption scheme vulnerabilities in distributed data protection middleware. It proposes and implements various attack methods and defense measures. Through experiments on real-world datasets, the project validates the vulnerabilities of existing encryption schemes in attribute recovery, frequency analysis, and other aspects, and proposes improved attack algorithms and corresponding defense strategies.

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
│   └── PotenialAttackUsingAuxiliary.py     # Potential auxiliary data attack
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
├── functions.py                # Core utility function library
└── README.md                   # Project documentation
```

## Core Features

### 1. Attack Schemes

#### 1.1 Attribute Recovery Attack
- **Files**: `NKW15andSSESchemes/AttributeRecoverAttack.py`, `Ours/AttributeRecoveryAttack.py`
- **Functionality**: Infers the mapping between ciphertext columns and plaintext columns by analyzing features such as column element frequency and unique value counts
- **Features**: 
  - Supports dynamic column mapping inference without prior knowledge of ciphertext headers
  - Based on Euclidean distance and feature vector similarity matching
  - Applicable to multiple encryption schemes: OPE, DET, SSE

#### 1.2 Frequency Analysis Attack
- **Files**: `NKW15andSSESchemes/FrequencyAnalysisAttack.py`, `Ours/EnhancedFrequencyAnalysisAttack.py`
- **Functionality**: Performs DET attacks using column element frequency distributions
- **Features**:
  - Uses Hungarian algorithm for optimal mapping matching
  - Supports dual matching based on frequency and cumulative distribution function (CDF)
  - Enhanced version supports auxiliary data attacks

#### 1.3 Cumulative Attack
- **Files**: `NKW15andSSESchemes/CumulativeAttack.py`, `Ours/EnhancedCumulativeAttack.py`
- **Functionality**: Gradually infers more plaintext by accumulating recovered information
- **Features**:
  - Based on co-occurrence frequency and graph algorithms
  - Supports multi-round iterative optimization
  - Enhanced version introduces auxiliary data to improve attack effectiveness

#### 1.4 SSE Attack (Searchable Symmetric Encryption)
- **Files**: `NKW15andSSESchemes/Oya2021_SSE.py`, `NKW15andSSESchemes/NWX24_SSE.py`
- **Functionality**: Attacks against searchable encryption schemes
- **Features**:
  - Utilizes query frequency and document frequency
  - Based on cost matrix and Hungarian algorithm
  - Supports external knowledge base assistance

### 2. Defense Measures

#### 2.1 Obfuscation
- **Files**: `Countermeasures/Obfuscation.py`, `Countermeasures/Obfuscation_A1.py`
- **Functionality**: Obscures real data by adding fake records
- **Features**:
  - Supports different obfuscation ratios
  - Maintains data statistical properties
  - Can be used in combination with auxiliary data

#### 2.2 Padding
- **Files**: `Countermeasures/padding.py`, `Countermeasures/padding_A1.py`
- **Functionality**: Increases data volume by padding with fake records
- **Features**:
  - Dynamically adjusts padding ratio
  - Maintains data distribution characteristics
  - Supports auxiliary data attack scenarios

## Dataset Description

### PUDF (Patient Utilization Data Files)
- **Plaintext**: `dataset/2015.csv`
- **Ciphertext**: `dataset/PUDF_base1_4q2018.csv`, `dataset/PUDF_base1_4q2019.csv`
- **Download Link**: [NYC Health + Hospitals Patient Utilization Data](https://www1.nyc.gov/site/hhc/data-transparency.page)
- **Main Fields**: Age, Gender, Risk, Admission Type, Race, Hospital, Principal Diagnosis
- **Encryption Types**: OPE, DET, SSE

### ACS (American Community Survey)
- **Plaintext**: `dataset/usa_2_plain.csv` (needs to be prepared)
- **Ciphertext**: `dataset/usa_2_cipher.csv` (needs to be prepared)
- **Download Link**: [U.S. Census Bureau American Community Survey](https://www.census.gov/programs-surveys/acs/data)
- **Main Fields**: VALUEH, AGE, SEX, MARST, RACE, EDUC, OCC, IND
- **Encryption Types**: OPE, DET, SSE

### Alzheimer
- **Plaintext**: `dataset/Alzheimer_plain.csv`
- **Ciphertext**: `dataset/Alzheimer_cipher.csv`
- **Download Link**: [CDC Alzheimer's Disease and Healthy Aging Data](https://www.cdc.gov/aging/agingdata/alzheimers-data-portal.html)
- **Main Fields**: YearStart, LocationAbbr, Stratification2, Class, DataValueTypeID, Topic, Low_Confidence_Limit, High_Confidence_Limit
- **Encryption Types**: OPE, DET, SSE

## Dataset Download Instructions

### 1. PUDF Dataset
```bash
# Visit the NYC Health + Hospitals website
# Download the Patient Utilization Data Files for the desired years
# Extract and place the CSV files in the dataset/ directory
```

### 2. ACS Dataset
```bash
# Visit the U.S. Census Bureau ACS data portal
# Download the Public Use Microdata Sample (PUMS) files
# Process and format according to your research requirements
# Place the processed files in the dataset/ directory
```

### 3. Alzheimer Dataset
```bash
# Visit the CDC Alzheimer's Disease and Healthy Aging Data Portal
# Download the Alzheimer's disease-related data
# Extract and place the CSV files in the dataset/ directory
```


## Experimental Setup

### Data Scale
Experiments are conducted at different data scales, ranging from 500 records to 113,657 records:
```python
base = [500, 11816, 23132, 34448, 45764, 57080, 68396, 79712, 91028, 102344, 113657]
```

### Number of Experiments
- Each data scale runs 50 experiments
- Calculate the average number of recovered keywords and success rate

### Evaluation Metrics
- **Keyword Count**: Total number of unique values in the dataset
- **Successfully Recovered Keywords**: Number of plaintext values successfully recovered by the attack
- **Recovery Rate**: Successfully recovered keywords / Total keywords

## Usage

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Running Attack Experiments

#### 2.1 Using Our Scheme
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

#### 2.2 Using Jigsaw Scheme
```bash
# PUDF 2010, 2018, 2019 dataset
python otherdataset/otherDataset_jigsaw/2018.py

# ACS dataset
python otherdataset/otherDataset_jigsaw/ACS.py

# Alzheimer dataset
python otherdataset/otherDataset_jigsaw/Alzheimer.py
```

#### 2.3 Using Oya 2021 Scheme
```bash
# PUDF 2010, 2018, 2019 dataset
python otherdataset/otherDataset_OK21/2018.py

# ACS dataset
python otherdataset/otherDataset_OK21/ACS.py

# Alzheimer dataset
python otherdataset/otherDataset_OK21/Alzheimer.py
```

### 3. Running Defense Experiments
```bash
# Obfuscation defense
python Countermeasures/Obfuscation.py

# Padding defense
python Countermeasures/padding.py
```

## License

This project is for academic research use only.

## Contact

For questions or suggestions, please contact the project maintainers.

## Changelog

### v1.0 (2025)
- Initial release
- Implemented multiple attack schemes
- Added defense measures
- Completed experimental validation on multiple datasets
