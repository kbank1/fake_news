import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from taxifare.ml_logic.registry import load_model

app = FastAPI()
app.state.model = pickle.load(open("pipeline.pkl","rb"))

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

    X_pred = pd.Series(text)

    pipe = app.state.model
    assert pipe is not None

    #X_processed = preprocess_features(X_pred) ADD AFTER PUTTING PREPROCESSOR IN PIPE

    prediction = pipe.predict(X_pred)[0]

    if prediction == 0:
        result = 'True'
    elif prediction == 1:
        result = 'Fake'
    else:
        result = 'ERROR'

    return {'Based on our current state-of-the-art algorithm, we predict that this text contains information that is': result}

@app.get("/")
def root():
    return {'Greeting': 'Hello, this is your final project'}

@app.get("/pipe")
def pipe():
    pipe = app.state.model
    assert pipe is not None

    return {'These are the pipeline parameters': list(pipe.get_params())}
