import pickle
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping

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

    # Set parameters
    max_features = 40000  # Maximum number of words to get out of our data
    max_len = 300  # Maximum sequence length
    embedding_dim = 50  # Dimensionality of word embeddings

    #tokenizing
    tk = Tokenizer(num_words=40000)
    tk.fit_on_texts(X_train)

    #tokenization
    X_train_token = tk.texts_to_sequences(X_train)
    X_test_token=tk.texts_to_sequences(X_test)

    # Pad the inputs to a fixed length
    X_train_pad = pad_sequences(X_train_token,  padding='pre', maxlen=max_len)
    X_test_pad=pad_sequences(X_test_token,  padding='pre', maxlen=max_len)

    # Build the model
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim))
    model.add(LSTM(16))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    metrics=['accuracy', 'Precision', 'Recall' ]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    # Define Early Stopping
    early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Stop when validation Recall stops improving
    patience=5,          # Wait for 5 epochs without improvement before stopping
    mode="max",
    restore_best_weights=True  # Restore the best model weights after stopping
    )

    # Train the model
    model.fit(X_train_pad, y_train, batch_size=128, epochs=20, validation_data=(X_test_pad, y_test), callbacks=[early_stopping])

    # Evaluate the model
    score=model.evaluate(X_test_pad, y_test, return_dict=True)
    accuracy=score["accuracy"]

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("tk.pkl", "wb") as file:
        pickle.dump(tk, file)

    print(f"✅ Model created with accuracy of: {round(accuracy,2)}")
