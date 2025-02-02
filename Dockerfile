FROM python:3.10.6-buster
COPY fakenews /fakenews
COPY api /api
COPY pipeline.pkl /pipeline.pkl
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
