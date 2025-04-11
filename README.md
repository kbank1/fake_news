# Fact or Fiction? Using Language to Spot Fake News

## Introduction
This repository contains the code and resources for detecting fake news using natural language processing and deep learning. With the rise of misinformation—especially around sensitive events like elections—this project aims to develop a tool that identifies fake news articles by analyzing linguistic patterns in text.

## Dataset
We combined two diverse datasets to improve the model's robustness and generalizability. These datasets consist of labeled real and fake news articles, allowing the model to learn distinctive language structures and stylistic cues associated with misinformation.
We included real and fake news articles from Fakenewsnet, Fakenewscorpus and the Guardian. Combined, our final dataset consists of 14,000 real and fake news articles (7,000 each).

## Methodology
Data Preprocessing: Text cleaning, tokenization, and vectorization to convert raw text into model-readable format.

Exploratory Data Analysis (EDA): Investigating word frequency, sentiment, and stylistic differences between real and fake news.

Model Building: We employed Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, due to their ability to capture sequential dependencies in text.

Model Evaluation: The model's performance was assessed using standard metrics: accuracy, precision, recall, and F1-score on a held-out test set.

## Results
The model successfully detects patterns in language that correlate with misinformation. While it performs well on the test set with an prediction accuracy of 85%, its reliability on real-world examples beyond our test set requires further validation. Our tool forms a foundational component for broader misinformation detection systems.

## Acknowledgments
Thanks to the creators and contributors of the publicly available fake news datasets used in this project.

Special thanks to the Le Wagon community for its support.

Check out our demo here: https://fakenews-1911.streamlit.app/

