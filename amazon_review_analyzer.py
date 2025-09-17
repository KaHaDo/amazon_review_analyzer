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
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# Load config file and global variables
from config import load_config

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

# Define function to compute coherence
def compute_coherence_values(vectorizer, X, dictionary, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    feature_names = vectorizer.get_feature_names_out()

    for num_topics in range(start, limit, step):
        lda_model = LatentDirichletAllocation(n_components=num_topics, n_jobs=-1, random_state=123)
        lda_model.fit(X)
        model_list.append(lda_model)

        topics = []
        for topic_idx, topic_weights in enumerate(lda_model.components_):
            top_words_idx = topic_weights.argsort()[:-15 - 1:-1]
            topics.append([feature_names[i] for i in top_words_idx])

        coherence_model = CoherenceModel(model=None, topics=topics, texts=texts, dictionary=dictionary, coherence='c_v',processes=-1)
        score = coherence_model.get_coherence()
        coherence_values.append(score)
        print(f"  - Topics: {num_topics}, Coherence Score: {round(score, 4)}")

    return model_list, coherence_values

# General Function to extract topics
def get_topics(model, feature_names, num_top_words):
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topics_dict[f'Topic {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
    return pd.DataFrame.from_dict(topics_dict, orient='index').transpose()

# Start of the Main Program
if __name__ == '__main__':

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
    print('\nStarting Bag-of-Words (BoW) transformation:')

    # Load the cleaned file
    df = pd.read_csv(output_path, na_filter=False)

    print(f"\nAfter loading in BoW test - Number of missing values in 'review_body': {df['review_body'].isna().sum()}")
    print(f'\nTotal number of rows: {len(df)}')

    # Initialize CountVectorizer

    bow_vectorizer = CountVectorizer(max_features=5000, min_df=5, max_df=0.9)

    # Apply BoW vectorization to the reviews
    X_bow = bow_vectorizer.fit_transform(df['review_body'])

    # Print the dimensions of the resulting sparse matrix
    print(f'\nShape of the BoW matrix: {X_bow.shape}')  # (Number of documents, Number of unique words)

    # Check vocabulary for unwanted characters
    vocabulary = bow_vectorizer.get_feature_names_out()
    filtered_vocabulary = [word for word in vocabulary if word.isalpha()]

    # Display the first 20 words from the cleaned vocabulary (Just to check)
    print('\nFirst 20 words from the vocabulary (Just to check): ' + ', '.join(filtered_vocabulary[:20]))

    # Coherence Score Calculation
    print('\nStarting Coherence Score Calculation to find optimal number of topics (this may take some time):')
    print()

    # Prepare data for Gensim
    tokenized_texts = [review.split() for review in df['review_body'].dropna()]
    id2word = Dictionary(tokenized_texts)

    # Calculate coherence scores for a range of topics
    start, limit, step = 5, 101, 5
    model_list, coherence_values = compute_coherence_values(
        vectorizer=bow_vectorizer, X=X_bow, dictionary=id2word, texts=tokenized_texts,
        start=start, limit=limit, step=step
    )

    # Visualize the results
    x = range(start, limit, step)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o')
    plt.title('LDA Coherence Score by Number of Topics')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.xticks(x)
    plt.grid(True)

    plot_output_path = config['PLOT_OUTPUT_PATH']
    plt.savefig(plot_output_path)
    print(f"\nCoherence plot saved to: {plot_output_path}")

    # Find the optimal number of topics
    best_score_index = np.argmax(coherence_values)
    optimal_num_topics = x[best_score_index]
    optimal_lda_model = model_list[best_score_index]

    print(f'\nOptimal number of topics is k={optimal_num_topics} with a Coherence Score of {max(coherence_values):.4f}')
    print('Coherence Score calculation completed!\n')

    # Starting tf-idf transformation
    print('\nStarting Term Frequency-Inverse Document Frequency (TF-IDF) transformation:')

    # Load the cleaned file
    df = pd.read_csv(output_path, na_filter=False)

    print(
        f"\nAfter loading in tf-idf test - Number of missing values in 'review_body': {df['review_body'].isna().sum()}")
    print(f'\nTotal number of rows: {len(df)}')

    # tf-idf Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.9)
    X_tfidf = tfidf_vectorizer.fit_transform(df['review_body'])

    # Output tf-idf matrix dimensions
    print(f'\nShape of the tf-idf matrix: {X_tfidf.shape}')  # (Number of documents, Number of unique words)

    # Display the first 20 words from the vocabulary (Just to check)
    print('\nFirst 20 words from the vocabulary (Just to check): ' + ', '.join(tfidf_vectorizer.get_feature_names_out()[:20]))

    # Maximum, minimum, and average tf-idf values
    print('\nOutput of tf-idf values:')
    print(f' - Max tf-idf value: {X_tfidf.max()}')
    print(f' - Min tf-idf value: {X_tfidf.min()}')
    print(f' - Average tf-idf value: {np.mean(X_tfidf):.3f}\n')

    # Latent Dirichlet Allocation (LDA)
    print(f'Starting Latent Dirichlet Allocation with k={optimal_num_topics} topics (This may take some time)')

    # Set the number of topics to the optimal value found by the coherence score
    #num_topics = optimal_num_topics

    # Train the LDA model
    # random_state=123 ensures reproducible results (same topics every run)
    lda_model = LatentDirichletAllocation(n_components=optimal_num_topics, random_state=123)
    lda_topics = lda_model.fit_transform(X_bow)  # X_bow is the BoW matrix

    df_lda_topics = get_topics(lda_model, bow_vectorizer.get_feature_names_out(), 15)

    # Print the topics as a formatted table
    print('\nExtracted topics from reviews using LDA:\n')
    print(df_lda_topics.to_string(index=False))

    # Save the LDA topics as a CSV file
    lda_output_path = config['LDA_OUTPUT_PATH']
    df_lda_topics.to_csv(lda_output_path, index=False)
    print(f'\nLDA topics saved as CSV at: {lda_output_path}\n')

# Starting Latent Semantic Analysis (LSA)
print('Starting Latent Semantic Analysis (this may take some time):')

# Train LSA model using Truncated SVD - Find optimal number of topics by using Scree Plot
# random_state=123 ensures reproducible results (same topics every run)
lsa_model = TruncatedSVD(n_components=100, random_state=123)
lsa_model.fit(X_tfidf)
explained_variance = lsa_model.explained_variance_ratio_

plt.figure(figsize=(12, 6))
plt.plot(range(1, 101), explained_variance, marker='o', linestyle='--')
plt.title('LSA Plot')
plt.xlabel("Number of Topics (Components)")
plt.ylabel("Explained Variance per Topic")
plt.grid(True)
lsa_plot_path = config['LSA_PLOT_PATH']
plt.savefig(lsa_plot_path)
print(f"\nLSA Scree Plot saved to: {lsa_plot_path}")

# Set optimal number of topics based on the LSA Scree Plot
optimal_num_topics_lsa = 30

print(f'\nTraining final LSA model with k={optimal_num_topics_lsa} topics (determined by Scree Plot)')

# Train a NEW, final LSA model with the optimal number of topics
final_lsa_model = TruncatedSVD(n_components=optimal_num_topics_lsa, random_state=123)
final_lsa_model.fit(X_tfidf)

# Extract topics from final model
df_lsa_topics = get_topics(final_lsa_model, tfidf_vectorizer.get_feature_names_out(), 15)

# Print the topics as a formatted table
print('\nExtracted topics from reviews using LSA:\n')
print(df_lsa_topics.to_string(index=False))

# Save the LSA topics as a CSV file
lsa_output_path = config['LSA_OUTPUT_PATH']
df_lsa_topics.to_csv(lsa_output_path, index=False)
print(f'\nLSA topics saved as CSV at: {lsa_output_path}\n')