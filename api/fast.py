import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fakenews.model import create_model
# from taxifare.ml_logic.registry import load_model

app = FastAPI()
#app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
"""
@app.get("/predict")
def predict(
        pickup_datetime: str,  # 2014-07-06 19:18:00
        pickup_longitude: float,    # -73.950655
        pickup_latitude: float,     # 40.783282
        dropoff_longitude: float,   # -73.984365
        dropoff_latitude: float,    # 40.769802
        passenger_count: int
    ):      # 1
# Add function explanation
    X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp(pickup_datetime, tz='US/Eastern')],
        pickup_longitude=[float(pickup_longitude)],
        pickup_latitude=[float(pickup_latitude)],
        dropoff_longitude=[float(dropoff_longitude)],
        dropoff_latitude=[float(dropoff_latitude)],
        passenger_count=[int(passenger_count)],
    ))

    model = app.state.model
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = float(model.predict(X_processed))

    return {'fare': y_pred}
"""

@app.get("/")
def root():
    return {'greeting': 'Hello, this is your final project'}

@app.get("/test")
def test():
    pipe, score = create_model()
    return {'This is the score': score}
