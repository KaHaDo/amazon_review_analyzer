# Import modules
import os
import subprocess
import pandas as pd
import re
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

# Load CSV file
file_path = config['FILE_PATH']
df = pd.read_csv(file_path, low_memory=False, dtype={'vote': 'str'}, na_filter=False)

# Rename & check if columns exist
df = df.rename(columns={'summary': 'review_title', 'reviewText': 'review_body', 'rating': 'star_rating'})
if not {'review_title', 'review_body', 'star_rating'}.issubset(df.columns):
    print('\nMissing columns! Check the CSV file.')
    exit()

# Keep only relevant columns & convert data types
df = df[['review_title', 'review_body', 'star_rating']].copy()
df['star_rating'] = df['star_rating'].astype(int)
df['review_body'] = df['review_body'].str.lower()
df['review_title'] = df['review_title'].str.lower()

# Cleaning function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\b\d+\b', '', text)  # Remove whole numbers as standalone words
    text = re.sub(r'\d+', '', text)  # Remove all numbers, including in combinations like '100lb'
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Limit repeated letters to a maximum of 2 times
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = ' '.join([word for word in text.split() if 2 < len(word) < 15])  # Remove very short & long words
    return text.strip()  # Remove unnecessary spaces

df['review_body'] = df['review_body'].apply(clean_text)
df['review_title'] = df['review_title'].apply(clean_text)

# Remove empty rows
df = df[(df['review_body'].str.strip() != '') & (df['review_title'].str.strip() != '')]

# Save cleaned dataset
output_path = config['OUTPUT_PATH']
df.to_csv(output_path, index=False, encoding='utf-8')
print(f'\nCleaned dataset saved at: {output_path}')