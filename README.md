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

### Configuration Setup

After installing the dependencies, you’ll need to update *two files*:

•⁠  ⁠⁠ config.yaml.example
•⁠  ⁠⁠ kaggle.json.example

### Kaggle API Key

The program requires a valid Kaggle API key.  
See the [Kaggle API documentation](https://www.kaggle.com/docs/api) for instructions on how to obtain one.

In ⁠ config.yaml ⁠, set (All variables can be left default except KAGGLE_CONFIG_DIR)

•⁠  ⁠*⁠ KAGGLE_CONFIG_DIR ⁠* → The folder where ⁠ kaggle.json ⁠ is located.

### Key Variables in ⁠config.yaml ⁠

•⁠  ⁠*⁠ DATASET_NAME ⁠*  
    The Kaggle dataset to be analyzed.
    
•⁠  ⁠*⁠ DOWNLOAD_PATH ⁠*  
    Directory where the dataset will be stored.  
    (Relative to the project’s root directory.)
    
•⁠  ⁠*Other ⁠ *_PATH ⁠ variables*  
    Define where processed files should be saved.  
    (Also relative to the project’s root directory.)

After defining all variables rename the config files to config.yaml and kaggle.json.

### Run the project

```bash 
python3 amazon_review_analyzer.py
```
