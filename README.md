# Amazon Review Analyzer

A Python project to download, process, and analyze Amazon product reviews using machine learning models like LDA and LSA.

## Installation

### Clone the project

```bash 
git clone https://github.com/KaHaDo/amazon_review_analyzer.git
```
```bash
cd amazon_review_analyzer
```

### Create a new virtual environment 

```bash 
python3 -m venv .venv
```

### Activate the environment

```bash 
source .venv/bin/activate
```

### Install dependencies

```bash 
pip install -r requirements.txt
```

## Configuration Setup

After installing the dependencies, you need to set up the following files:

•⁠ `kaggle.json.example` – obtain your Kaggle API key from [Kaggle](https://www.kaggle.com/docs/api) and choose one of the following options:

  1. Copy the downloaded `kaggle.json` file directly into the project folder.
  2. Or, open `kaggle.json.example`, paste the contents of the downloaded file, and rename it to `kaggle.json`.

•⁠ `config.yaml.example` – modify as needed and rename to `config.yaml`. 
 
  Set the variable **KAGGLE_CONFIG_DIR** to the folder where `kaggle.json` is located.  
  All other variables can be left at their default values.

### Key Variables in ⁠config.yaml ⁠

•⁠  ⁠*⁠ DATASET_NAME ⁠*  
    The Kaggle dataset to be analyzed.
    
•⁠  ⁠*⁠ DOWNLOAD_PATH ⁠*  
    Directory where the dataset will be stored.  
    (Relative to the project’s root directory.)
    
•⁠  ⁠*Other ⁠ *_PATH ⁠ variables*  
    Define where processed files should be saved.  
    (Also relative to the project’s root directory.)

Make sure to rename the files after editing:
•⁠ `config.yaml.example` → `config.yaml`
•⁠ `kaggle.json.example` → `kaggle.json`

The program will **not run** if the files are not renamed.

### Run the project

```bash 
python3 amazon_review_analyzer.py
```
