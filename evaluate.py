from datasets import load_dataset,DatasetDict, Dataset
from transformers import pipeline,Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,multilabel_confusion_matrix,confusion_matrix
from typing import List, Dict, Any
import numpy as np


"""
Module for evaluating the pre-trained RoBERTa sentiment analysis model on the TweetEval dataset.

This script performs the following key operations:
1. Loads the TweetEval sentiment analysis test dataset
2. Initializes the pre-trained RoBERTa model from Hugging Face
3. Computes predictions on the test set
4. Evaluates model performance using standard metrics:
   - Accuracy
   - F1 Score
   - Precision
   - Recall
5. Calcualtes and print Confusion Matrix
"""
# loading the test dataset
dataset: DatasetDict = load_dataset("tweet_eval", "sentiment")
test_data: Dataset = dataset['test']

# loading the pre-trained mdoel from hugging face
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier: Pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# extract texts
texts: List[str] = test_data["text"]

# extract correspognding sentiments
ints_label_test: List[int] = test_data["label"]


# making the model predicts the sentiments
predictions: List[Dict[str, Any]] = classifier(texts)

# extract the sentiments part from the predictions
predicted_labels = [pred["label"] for pred in predictions]

# the model by default labels the pridictions LABEL_0 for negative, LABEL_1 for neutral and LABEL_2 for positive
label_to_int: Dict[str, int] = {
     "LABEL_0": 0,
     "LABEL_1": 1,
     "LABEL_2": 2
     }

# transform the labels into ints to match the test dataset
predicted_labels_int: List[int] = [label_to_int[label] for label in predicted_labels]


# compute confusion matrix
matrix: np.ndarray = confusion_matrix(ints_label_test, predicted_labels_int)


# compute metrics
accuracy: float = accuracy_score(ints_label_test, predicted_labels_int)
f1: float = f1_score(ints_label_test, predicted_labels_int, average="weighted")
precision: float = precision_score(ints_label_test, predicted_labels_int, average="weighted")
recall: float = recall_score(ints_label_test, predicted_labels_int, average="weighted")


# print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("Confusion Matrix")
print(matrix)


