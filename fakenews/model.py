import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from fakenews.data import load_data
from fakenews.preprocessor import preprocessing

def create_model():
    """
    Load training data.
    Create X/y variables and train/test split.
    Make pipeline with vectorizer and Naive Bayes model.
    Fit the model, display model score and return model.

    """

    data = load_data()

    X = data['text']
    y = data['fake']

    X = X.apply(preprocessing)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

    pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    with open("pipeline.pkl", "wb") as file:
        pickle.dump(pipe, file)

    print(f"✅ Model created with accuracy of: {round(score,2)}")
