import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import SMOTE
from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.layouts import column

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

    def _train_svm(self):
        svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        svm_grid = GridSearchCV(SVC(), svm_param_grid, refit=True, cv=5, verbose=2)
        svm_grid.fit(self.X_train, self.y_train)
        y_pred_svm = svm_grid.predict(self.X_test)
        print("Best SVM Parameters:", svm_grid.best_params_)
        classification_rep = classification_report(self.y_test, y_pred_svm)
        print("SVM Classification Report:")
        print(classification_rep)

        svm_scores = cross_val_score(svm_grid.best_estimator_, self.X_train, self.y_train, cv=5)
        print("SVM Cross-validation scores:", svm_scores)

        # Save results and visualize
        self._save_output("SVM", classification_rep, svm_scores)
        self._visualize_scores("SVM", self.y_test, y_pred_svm, svm_scores)

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
        self._train_svm()
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

main()