import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class SentimentAnalysis:
    def __init__(self, file_path):
        self._data = pd.read_csv(file_path)
        self._sid = SentimentIntensityAnalyzer()

    def __repr__(self):
        print(self._data)

    def preprocess(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]
        # Join words back to string
        clean_text = ' '.join(words)
        return clean_text
    
    def analyze_sentiment(self):
        clean_comments = []
        sentiments = []
        sentiment_labels = []

        for idx, comment in enumerate(self._data['Sentence']):
            if idx % 10000 == 0:
                print(f'Processing row: {idx}')
            clean_comment = self.preprocess(comment)
            clean_comments.append(clean_comment)
            sentiment = self._sid.polarity_scores(clean_comment)['compound']
            sentiments.append(sentiment)
            if sentiment > 0.05:
                sentiment_labels.append('Positive')
            elif sentiment < -0.05:
                sentiment_labels.append('Negative')
            else:
                sentiment_labels.append('Neutral')
        
        self._data['clean_comment'] = clean_comments
        self._data['sentiment'] = sentiments
        self._data['sentiment_label'] = sentiment_labels

    def save_results(self, output_path):
        self._data.to_csv(output_path, index=False)

    def print_sample(self, n=5):
        print(self._data[['Sentence', 'clean_comment', 'sentiment', 'sentiment_label']].head(n))

test = SentimentAnalysis("sentiment.csv")
test.analyze_sentiment()
test.print_sample()
test.save_results("test.csv")