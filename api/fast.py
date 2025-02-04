import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fakenews.preprocessor import preprocessing

app = FastAPI()
app.state.model = pickle.load(open("model.pkl","rb"))
app.state.tk = pickle.load(open("tk.pkl","rb"))

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(text: str):

    model = app.state.model
    assert model is not None

    max_len = 300

    X_pred = pd.Series(text)
    X_pred = X_pred.apply(preprocessing)
    X_pred_token = app.state.tk.texts_to_sequences(X_pred)
    X_pred_pad = pad_sequences(X_pred_token,  padding='pre', maxlen=max_len)

    prediction = model.predict(X_pred_pad)[0][0]

    if prediction < 0.5:
        result = 'True'
    elif prediction >= 0.5:
        result = 'Fake'
    else:
        result = 'ERROR'

    return {'Based on our current state-of-the-art algorithm, we predict that this text contains information that is': result}

@app.get("/")
def root():
    return {'Greeting': 'Hello, this is your final project'}

@app.get("/layers")
def layers():
    model = app.state.model
    assert model is not None

    return {'These are the layers of our model': [layer.name for layer in model.layers]}
