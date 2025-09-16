# Import modules
import os
import subprocess
import nltk
from nltk.corpus import stopwords

# Load config file and global variable
from config import load_config
config = load_config()

# Set Kaggle config (adjust the path!)
os.environ['KAGGLE_CONFIG_DIR'] = config['KAGGLE_CONFIG_DIR']

# Download stopwords only once!
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Download & extract Kaggle dataset
dataset_name = config['DATASET_NAME']
download_path = config['DOWNLOAD_PATH']
command = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', download_path, '--unzip']

try:
    subprocess.run(command, check=True)
    print(f'\nDownload successful! Files saved in: {download_path}')
except subprocess.CalledProcessError as e:
    print(f'\nError during download: {e}')