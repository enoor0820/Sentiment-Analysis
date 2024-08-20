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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# SentimentAnalysis class
class SentimentAnalysis:
    def __init__(self, file_path):
        self._data = pd.read_csv(file_path)
        self._stopwords = set(stopwords.words('english'))
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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
        print("SVM Classification Report:")
        print(classification_report(self.y_test, y_pred_svm))

        svm_scores = cross_val_score(svm_grid.best_estimator_, self.X_train, self.y_train, cv=5)
        print("SVM Cross-validation scores:", svm_scores)

    def _train_logistic_regression(self):
        lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 2000]}
        lr_grid = GridSearchCV(LogisticRegression(class_weight='balanced'), lr_param_grid, refit=True, cv=5, verbose=2)
        lr_grid.fit(self.X_train, self.y_train)
        y_pred_lr = lr_grid.predict(self.X_test)
        print("Best Logistic Regression Parameters:", lr_grid.best_params_)
        print("Logistic Regression Classification Report:")
        print(classification_report(self.y_test, y_pred_lr))

        lr_scores = cross_val_score(lr_grid.best_estimator_, self.X_train, self.y_train, cv=5)
        print("Logistic Regression Cross-validation scores:", lr_scores)

    def train_and_evaluate(self, X, y):
        self._scale_and_resample(X, y)
        self._train_svm()
        self._train_logistic_regression()
    
    # def train_and_evaluate(self, X, y):
    #     # Feature Scaling using MaxAbsScaler (suitable for sparse matrices)
    #     scaler = MaxAbsScaler()
    #     X_scaled = scaler.fit_transform(X)

    #     # Handle class imbalance using SMOTE
    #     smote = SMOTE()
    #     X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    #     # Train-test split
    #     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    #     # SVM Classifier with GridSearchCV
    #     svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    #     svm_grid = GridSearchCV(SVC(), svm_param_grid, refit=True, cv=5, verbose=2)
    #     svm_grid.fit(X_train, y_train)
    #     y_pred_svm = svm_grid.predict(X_test)
    #     print("Best SVM Parameters:", svm_grid.best_params_)
    #     print("SVM Classification Report:")
    #     print(classification_report(y_test, y_pred_svm))

    #     # Logistic Regression with Increased Iterations and Class Weight Balancing
    #     lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 2000]}
    #     lr_grid = GridSearchCV(LogisticRegression(class_weight='balanced'), lr_param_grid, refit=True, cv=5, verbose=2)
    #     lr_grid.fit(X_train, y_train)
    #     y_pred_lr = lr_grid.predict(X_test)
    #     print("Best Logistic Regression Parameters:", lr_grid.best_params_)
    #     print("Logistic Regression Classification Report:")
    #     print(classification_report(y_test, y_pred_lr))

    #     # Cross-validation scores for Logistic Regression
    #     lr_scores = cross_val_score(lr_grid.best_estimator_, X_resampled, y_resampled, cv=5)
    #     print("Logistic Regression Cross-validation scores:", lr_scores)

    #     # Cross-validation scores for SVM
    #     svm_scores = cross_val_score(svm_grid.best_estimator_, X_resampled, y_resampled, cv=5)
    #     print("SVM Cross-validation scores:", svm_scores)

analysis = SentimentAnalysis("sentiment.csv")

# Preprocess and prepare data
comments, sentiments = analysis.prepare_data()

# TF-IDF Vectorization
X, vectorizer = analysis.tfidf_vectorization(comments)

# Train and evaluate models
analysis.train_and_evaluate(X, sentiments)