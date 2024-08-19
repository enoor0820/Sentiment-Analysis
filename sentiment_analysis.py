import pandas as pd

class SentimentAnalysis:
    def __init__(self, file_path):
        self._data = pd.read_csv(file_path)

    def print(self):
        print(self._data)

    def preprocess(text):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]
        clean_text = ' '.join(words)
        return clean_text