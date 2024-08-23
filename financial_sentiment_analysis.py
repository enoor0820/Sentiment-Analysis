import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import SMOTE
from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.layouts import column
from collections import Counter
import matplotlib.pyplot as plt
import os


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# SentimentAnalysis class
class SentimentAnalysis:
    def __init__(self, file_path, output_file_path="output.txt"):
        self._data = pd.read_csv(file_path)
        self._stopwords = set(stopwords.words('english'))
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.output_file_path = output_file_path
        
        # Open the output file in write mode, clearing its contents if it exists
        open(self.output_file_path, 'w')

    def preprocess(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in self._stopwords]
        # Join words back to string
        clean_text = ' '.join(words)
        return clean_text
    
    def prepare_data(self):
        self._data['clean_comment'] = self._data['Sentence'].apply(self.preprocess)
        return self._data['clean_comment'], self._data['Sentiment']
    
    def tfidf_vectorization(self, comments, ngram_range=(1, 2), max_features=5000):
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        X = vectorizer.fit_transform(comments)
        return X, vectorizer
    
    def _scale_and_resample(self, X, y):
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    def _train_logistic_regression(self):
        lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 2000]}
        lr_grid = GridSearchCV(LogisticRegression(class_weight='balanced'), lr_param_grid, refit=True, cv=5, verbose=2)
        lr_grid.fit(self.X_train, self.y_train)
        y_pred_lr = lr_grid.predict(self.X_test)
        print("Best Logistic Regression Parameters:", lr_grid.best_params_)
        classification_rep = classification_report(self.y_test, y_pred_lr)
        print("Logistic Regression Classification Report:")
        print(classification_rep)

        lr_scores = cross_val_score(lr_grid.best_estimator_, self.X_train, self.y_train, cv=5)
        print("Logistic Regression Cross-validation scores:", lr_scores)

        # Save results and visualize
        self._save_output("Logistic Regression", classification_rep, lr_scores)
        self._visualize_scores("Logistic Regression", self.y_test, y_pred_lr, lr_scores)

    def train_and_evaluate(self, X, y):
        self._scale_and_resample(X, y)
        self._train_logistic_regression()

    def _save_output(self, model_name, classification_rep, cross_val_scores):
        with open(self.output_file_path, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write("Classification Report:\n")
            f.write(classification_rep + "\n")
            f.write("Cross-validation scores:\n")
            f.write(", ".join([f"{score:.4f}" for score in cross_val_scores]) + "\n")
            f.write(f"Mean Cross-validation score: {cross_val_scores.mean():.4f}\n\n")

    def _visualize_scores(self, model_name, y_test, y_pred, cross_val_scores):
        output_file(f"{model_name}_visualization.html")
        
        # F1 Scores by class
        report = classification_report(y_test, y_pred, output_dict=True)
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        p1 = figure(x_range=classes, title=f"{model_name} F1 Scores by Class", toolbar_location=None, tools="")
        p1.vbar(x=classes, top=f1_scores, width=0.9)
        p1.y_range.start = 0
        p1.xgrid.grid_line_color = None
        p1.yaxis.axis_label = "F1 Score"

        # Convert the range to a list for Bokeh to handle it properly
        folds = list(range(1, len(cross_val_scores) + 1))
        
        # Cross-validation scores
        p2 = figure(title=f"{model_name} Cross-validation Scores", toolbar_location=None, tools="")
        p2.line(folds, cross_val_scores, line_width=2)
        p2.circle(folds, cross_val_scores, size=10)
        p2.yaxis.axis_label = "Score"
        p2.xaxis.axis_label = "Fold"

        # Layout and save
        layout = column(p1, p2)
        save(layout)
        show(layout)
    
    def _create_common_words_df(self):
        # Analyze comments in support of the policy
        supportive_comments = self._data[self._data['Sentiment'] == 'positive']
        non_supportive_comments = self._data[self._data['Sentiment'] == 'negative']

        # Most common words in supportive comments
        supportive_words = ' '.join(supportive_comments['clean_comment']).split()
        common_supportive_words = Counter(supportive_words).most_common(50)

        # Most common words in non-supportive comments
        non_supportive_words = ' '.join(non_supportive_comments['clean_comment']).split()
        common_non_supportive_words = Counter(non_supportive_words).most_common(50)

        # Combine common words into a single DataFrame
        common_words = []
        for word, count in common_supportive_words:
            common_words.append({'word': word, 'count': count, 'supportive': 'True'})
        for word, count in common_non_supportive_words:
            common_words.append({'word': word, 'count': count, 'supportive': 'False'})

        common_words_df = pd.DataFrame(common_words)
        
        # Save common words to CSV file
        common_words_df.to_csv('common_words.csv', index=False)

        return common_supportive_words, common_non_supportive_words

    def _plot_common_words(self, common_supportive_words, common_non_supportive_words):
        # Plotting common words
        supportive_df = pd.DataFrame(common_supportive_words, columns=['word', 'count'])
        non_supportive_df = pd.DataFrame(common_non_supportive_words, columns=['word', 'count'])

        plt.figure(figsize=(12, 6))
        plt.bar(supportive_df['word'], supportive_df['count'], color='blue', alpha=0.7, label='Supportive')
        plt.bar(non_supportive_df['word'], non_supportive_df['count'], color='red', alpha=0.7, label='Non-Supportive')
        plt.title('Common Words in Comments')
        plt.xlabel('Words')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

    def analyze_patterns(self):
        # Create common words DataFrame
        common_supportive_words, common_non_supportive_words = self._create_common_words_df()
        
        # Plot common words
        self._plot_common_words(common_supportive_words, common_non_supportive_words)

    def sentiment_distribution(self):
        sentiment_counts = self._data['Sentiment'].value_counts()
        total = len(self._data)
        distribution = (sentiment_counts / total) * 100
        print("Sentiment Distribution:")
        print(distribution)

        with open(self.output_file_path, 'a') as f:
            f.write("Sentiment Distribution:\n")
            f.write(distribution.to_string() + "\n\n")

    def most_common_perspectives(self):
        positive_comments = self._data[self._data['Sentiment'] == 'positive']['clean_comment']
        negative_comments = self._data[self._data['Sentiment'] == 'negative']['clean_comment']

        positive_words = ' '.join(positive_comments).split()
        negative_words = ' '.join(negative_comments).split()

        common_positive = Counter(positive_words).most_common(10)
        common_negative = Counter(negative_words).most_common(10)

        print("Most Common Positive Perspectives:")
        print(common_positive)

        print("\nMost Common Negative Perspectives:")
        print(common_negative)

        with open(self.output_file_path, 'a') as f:
            f.write("Most Common Positive Perspectives:\n")
            for word, count in common_positive:
                f.write(f"{word}: {count}\n")

            f.write("\nMost Common Negative Perspectives:\n")
            for word, count in common_negative:
                f.write(f"{word}: {count}\n")

            f.write("\n")

    def main_themes(self):
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(self._data['clean_comment'])
        features = vectorizer.get_feature_names_out()
        word_count = X.toarray().sum(axis=0)
        themes = Counter(dict(zip(features, word_count))).most_common(10)
        
        print("Main Themes:")
        print(themes)

        with open(self.output_file_path, 'a') as f:
            f.write("Main Themes:\n")
            for theme, count in themes:
                f.write(f"{theme}: {count}\n")
            f.write("\n")

    def example_perspectives(self):
        positive_example = self._data[self._data['Sentiment'] == 'positive']['Sentence'].iloc[0]
        negative_example = self._data[self._data['Sentiment'] == 'negative']['Sentence'].iloc[0]
        neutral_example = self._data[self._data['Sentiment'] == 'neutral']['Sentence'].iloc[0]

        print("Example Supportive Perspective:")
        print(positive_example)

        print("\nExample Opposed Perspective:")
        print(negative_example)

        print("\nExample Moderate Perspective:")
        print(neutral_example)

        with open(self.output_file_path, 'a') as f:
            f.write("Example Supportive Perspective:\n")
            f.write(positive_example + "\n\n")

            f.write("Example Opposed Perspective:\n")
            f.write(negative_example + "\n\n")

            f.write("Example Moderate Perspective:\n")
            f.write(neutral_example + "\n\n")

    def practical_suggestions(self):
        suggestions = self._data[self._data['clean_comment'].str.contains('should|recommend|suggest|need', regex=True)]['Sentence']
        print("Practical Suggestions and Recommendations:")
        for suggestion in suggestions:
            print(f"- {suggestion}")

        with open(self.output_file_path, 'a') as f:
            f.write("Practical Suggestions and Recommendations:\n")
            for suggestion in suggestions:
                f.write(f"- {suggestion}\n")
            f.write("\n")


def main():
    # Example usage
    file_path = "sentiment.csv"  # Replace with your file path
    output_file_path = "output.txt"
    
    analysis = SentimentAnalysis(file_path, output_file_path)

    # Preprocess and prepare data
    comments, sentiments = analysis.prepare_data()

    # TF-IDF Vectorization
    X, vectorizer = analysis.tfidf_vectorization(comments)

    # Train and evaluate models
    analysis.train_and_evaluate(X, sentiments)
    
    # Analyze and save results to output file
    analysis.analyze_patterns()
    analysis.sentiment_distribution()
    analysis.most_common_perspectives()
    analysis.main_themes()
    analysis.example_perspectives()
    analysis.practical_suggestions()

main()
