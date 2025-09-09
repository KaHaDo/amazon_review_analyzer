## Installation

### Clone the project

git clone https://github.com/KaHaDo/amazon_review_analyzer.git

### Create a new virtual environment 

python3 -m venv ./amazon_review_analyzer/.venv

### Activate the environment

source ./amazon_review_analyzer/.venv/bin/activate

### Install dependencies

pip install -r requirements.txt

### Run the project

### Configuration Setup

After installing the dependencies, you’ll need to update *two files*:

•⁠  ⁠⁠ config.yaml.example
•⁠  ⁠⁠ kaggle.json.example

### Kaggle API Key

The program requires a valid Kaggle API key to function.  
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

After defining all variables rename the Config-files to config.yaml and kaggle.json. The program can be started as follows:

python3 07_LDA_LSA.py (final script amazon_review_analyzer.py - to be changed)

