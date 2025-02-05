FROM python:3.10.6-buster
COPY fakenews /fakenews
COPY api /api
COPY model.pkl /model.pkl
COPY tk.pkl /tk.pkl
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
