# Import modules
import os
import subprocess
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

# Load config file and global variables
from config import load_config
config = load_config()

# Set Kaggle config
os.environ['KAGGLE_CONFIG_DIR'] = config['KAGGLE_CONFIG_DIR']

# Download stopwords (only once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Download & extract Kaggle dataset
dataset_name = config['DATASET_NAME']
download_path = config['DOWNLOAD_PATH']
command = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', download_path, '--unzip']

try:
    subprocess.run(command, check=True)
    print(f'\nDownload successful! Original dataset saved at: {download_path}')
except subprocess.CalledProcessError as e:
    print(f'\nError during download: {e}')

# Load CSV file
file_path = config['FILE_PATH']
df = pd.read_csv(file_path, low_memory=False, dtype={'vote': 'str'}, na_filter=False)

# Rename & check if columns exist
df = df.rename(columns={'summary': 'review_title', 'reviewText': 'review_body', 'rating': 'star_rating'})
if not {'review_title', 'review_body', 'star_rating'}.issubset(df.columns):
    print('\nMissing columns! Check the CSV file.\n')
    exit()

# Keep only relevant columns & convert data types
df = df[['review_title', 'review_body', 'star_rating']].copy()
df['star_rating'] = df['star_rating'].astype(int)
df['review_body'] = df['review_body'].str.lower()
df['review_title'] = df['review_title'].str.lower()

# Dataset cleaning
print('\nStarting dataset cleaning process!')

# Cleaning function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\b\d+\b', '', text)  # Remove whole numbers as standalone words
    text = re.sub(r'\d+', '', text)  # Remove all numbers, including combinations like '100lb'
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Limit repeated letters to a maximum of 2
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = ' '.join([word for word in text.split() if 2 < len(word) < 15])  # Remove very short & long words
    return text.strip()  # Remove unnecessary spaces

df['review_body'] = df['review_body'].apply(clean_text)
df['review_title'] = df['review_title'].apply(clean_text)

# Remove empty rows
df = df[(df['review_body'].str.strip() != '') & (df['review_title'].str.strip() != '')]

# Cleaning process completed
print('\nCleaning process completed!')

# Save cleaned Dataset as CSV
output_path = config['OUTPUT_PATH']
df.to_csv(output_path, index=False, encoding='utf-8')
print(f'\nCleaned dataset saved at: {output_path}')

# Starting bag-of-words (BoW) transformation
print('\nStarting bag-of-words (BoW) transformation:')

# Load the cleaned file
df = pd.read_csv(output_path, na_filter=False)

print(f"\nAfter loading in BoW test - Number of missing values in 'review_body': {df['review_body'].isna().sum()}")
print(f'\nTotal number of rows: {len(df)}')

# Initialize CountVectorizer
bow_vectorizer = CountVectorizer(max_features=250)  # Limit to 250 words

# Apply BoW vectorization to the reviews
X_bow = bow_vectorizer.fit_transform(df['review_body'])

# Print the dimensions of the resulting sparse matrix
print(f'\nShape of the BoW matrix: {X_bow.shape}')  # (Number of documents, Number of unique words)

# Check vocabulary for unwanted characters
vocabulary = bow_vectorizer.get_feature_names_out()
filtered_vocabulary = [word for word in vocabulary if word.isalpha()]

# Display the first 20 words from the cleaned vocabulary
print('\nFirst 20 words from the vocabulary: ' + ', '.join(filtered_vocabulary[:20]))

# Save the BoW matrix as a CSV file
df_bow = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
bow_output_path = config['BOW_OUTPUT_PATH']
df_bow.to_csv(bow_output_path, index=False)

print(f'\nBag-of-Words (BoW) transformation completed! Matrix saved at: {bow_output_path}')

# Starting tf-idf transformation
print('\nStarting tf-idf transformation:')

# Load the cleaned file
df = pd.read_csv(output_path, na_filter=False)

print(f"\nAfter loading in tf-idf test - Number of missing values in 'review_body': {df['review_body'].isna().sum()}")
print(f'\nTotal number of rows: {len(df)}')

# tf-idf Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=250)  # Limit to 250 words
X_tfidf = tfidf_vectorizer.fit_transform(df['review_body'])

# Output tf-idf matrix dimensions
print(f'\nShape of the tf-idf matrix: {X_tfidf.shape}')  # (Number of documents, Number of unique words)

# Display the first 20 words from the vocabulary
print('\nFirst 20 words from the vocabulary: ' + ', '.join(tfidf_vectorizer.get_feature_names_out()[:20]))

# Maximum, minimum, and average tf-idf values
print('\nOutput of tf-idf values:')
print(f' - Max tf-idf value: {X_tfidf.max()}')
print(f' - Min tf-idf value: {X_tfidf.min()}')
print(f' - Average tf-idf value: {np.mean(X_tfidf):.3f}')

# Save the tf-idf matrix as a CSV file
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_output_path = config['TFIDF_OUTPUT_PATH']
df_tfidf.to_csv(tfidf_output_path, index=False, float_format='%.6f')

print(f'\nTF-IDF transformation completed! Matrix saved at: {tfidf_output_path}\n')

# Latent Dirichlet Allocation (LDA)
print('Starting Latent Dirichlet Allocation (this may take some time):')

# Define number of topics
num_topics = 25

# Train the LDA model
# random_state=123 ensures reproducible results (same topics every run)
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=123)
lda_topics = lda_model.fit_transform(X_bow)  # X_bow is the BoW matrix

# Function to extract top words for each topic
def get_lda_topics(model, feature_names, num_top_words):
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topics_dict[f'Topic {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
    return pd.DataFrame.from_dict(topics_dict, orient='index').transpose()

# Extract topics as DataFrame
df_lda_topics = get_lda_topics(lda_model, bow_vectorizer.get_feature_names_out(), 15)

# Print the topics as a formatted table
print('\nExtracted topics from reviews using LDA:\n')
print(df_lda_topics.to_string(index=False))

# Save the LDA topics as a CSV file
lda_output_path = config['LDA_OUTPUT_PATH']
df_lda_topics.to_csv(lda_output_path, index=False)
print(f'\nLDA topics saved as CSV at: {lda_output_path}\n')

# Starting Latent Semantic Analysis (LSA)
print('Starting Latent Semantic Analysis (this may take some time):')

# Define number of topics
num_topics = 25

# Train LSA model using Truncated SVD
# random_state=123 ensures reproducible results (same topics every run)
lsa_model = TruncatedSVD(n_components=num_topics, random_state=123)
lsa_matrix = lsa_model.fit_transform(X_tfidf)  # X_tfidf is the tf-idf matrix

# Function to extract top words for each topic
def get_lsa_topics(model, feature_names, num_top_words):
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topics_dict[f'Topic {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
    return pd.DataFrame.from_dict(topics_dict, orient='index').transpose()

# Extract topics as DataFrame
df_lsa_topics = get_lsa_topics(lsa_model, tfidf_vectorizer.get_feature_names_out(), 15)

# Print the topics as a formatted table
print('\nExtracted topics from reviews using LSA:\n')
print(df_lsa_topics.to_string(index=False))

# Save the LSA topics as a CSV file
lsa_output_path = config['LSA_OUTPUT_PATH']
df_lsa_topics.to_csv(lsa_output_path, index=False)
print(f'\nLSA topics saved as CSV at: {lsa_output_path}\n')