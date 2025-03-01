from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,multilabel_confusion_matrix,confusion_matrix
import numpy as np

"""
this module loads tweet_eval test dataset to evaluate the pre-trained RoBERTa sentiment analysis AI model
"""
# loading the test dataset
dataset = load_dataset("tweet_eval", "sentiment")
test_data = dataset['test']

# loading the pre-trained mdoel from hugging face
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# extract texts
texts = test_data["text"]

# extract correspognding sentiments
ints_label_test = test_data["label"]


# making the model predicts the sentiments
predictions = classifier(texts)

# extract the sentiments part from the predictions
predicted_labels = [pred["label"] for pred in predictions]

# the model by default labels the pridictions LABEL_0 for negative, LABEL_1 for neutral and LABEL_2 for positive
label_to_int = {
     "LABEL_0": 0,
     "LABEL_1": 1,
     "LABEL_2": 2
     }

# transform the labels into ints to match the test dataset
predicted_labels_int = [label_to_int[label] for label in predicted_labels]


# compute confusion matrix
matrix = confusion_matrix(ints_label_test, predicted_labels_int)


# compute metrics
accuracy = accuracy_score(ints_label_test, predicted_labels_int)
f1 = f1_score(ints_label_test, predicted_labels_int, average="weighted")
precision = precision_score(ints_label_test, predicted_labels_int, average="weighted")
recall = recall_score(ints_label_test, predicted_labels_int, average="weighted")


# print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("==============================")
print(matrix)


