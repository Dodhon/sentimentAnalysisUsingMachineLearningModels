####### Text preprocessing techniques
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('Tweets.csv')

# Define the preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove user mentions and URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    white_list = ["not", "no", "won't", "isn't", "couldn't", "wasn't", "didn't", "shouldn't", 
                  "hasn't", "wouldn't", "haven't", "weren't", "hadn't", "shan't", "doesn't",
                  "mightn't", "mustn't", "needn't", "don't", "aren't", "won't"]
    
    words = text.split()
    text = ' '.join([t for t in words if (t not in stop_words or t in white_list)])
    
    # Remove punctuations
    text = ''.join([t for t in text if t not in string.punctuation])
    
    # Remove numeric numbers
    text = ''.join([t for t in text if not t.isdigit()])
    
    # Stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(t) for t in word_tokenize(text)])
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    text = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(t, get_wordnet_pos(pos)) for t, pos in pos_tag(text)])
    
    return text

# Apply the preprocessing function to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)

# Export the DataFrame to a new CSV file
output_path = 'Tweets_processed.csv'
df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
