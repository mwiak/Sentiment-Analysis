import tkinter as tk
from typing import List, Dict, Any
from tkinter import filedialog, scrolledtext
from transformers import pipeline,Pipeline
import pandas as pd
import json
from collections import Counter

"""
A GUI application for sentiment analysis using Hugging Face's transformer model.

This module provides a graphical user interface (GUI) to perform sentiment analysis
on either direct text input or data from CSV/JSON files. It utilizes the
'twitter-roberta-base-sentiment' model from Hugging Face to predict sentiments and
displays results including sentiment labels, confidence scores, and file summaries.
"""

# load sentiment analysis model pipeline from Hugging Face
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier: Pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


# the model by default labels the pridictions LABEL_0 for negative, LABEL_1 for neutral and LABEL_2 for positive
label_to_sentiment: Dict[str,str] = {
     "LABEL_0": 'negative',
     "LABEL_1": 'neutral',
     "LABEL_2": 'positive'
     }


def analyze_text() -> None:
    """
    Analyze sentiment of text entered in the GUI text box.

    Retrieves text from the input widget, processes it through the sentiment analysis
    model, and displays the predicted sentiment along with its confidence score.
    Updates the result in the GUI's output label.
    """
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        result: List[Dict[str, Any]] = classifier(text)[0]
        output_label.config(text=f"Sentiment: {label_to_sentiment[result['label']]} (Score: {result['score']:.2f})")

def analyze_file() -> None:
    """
    Perform sentiment analysis on data from a selected CSV or JSON file.

    Prompts the user to select a file, reads the data, and processes all text entries
    through the sentiment analysis model. Displays a summary of sentiment distribution
    including counts and percentages in the GUI's scrollable text widget. Handles
    common file processing errors and displays error messages when encountered.

    Supported File Formats:
        - CSV: Reads text from the first column.
        - JSON: Expects a list of strings in the JSON file.

    Raises:
        Exception: Generic exception handling for file reading/processing errors,
                   with error details displayed in the GUI.
    """
    
    # prompt user to select a JSON or CSV file
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"),("JSON Files", "*.JSON")])
    if file_path:
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                texts = df.iloc[:, 0].astype(str).tolist()
            elif file_path.endswith(".JSON"):
                with open(file_path, "r", encoding="utf-8") as f:
                 texts = json.load(f)   
           
            # perform sentiment analysis on several text entries
            results: List[Dict[str, any]] = classifier(texts)
            # delete the prevoius summary if exists
            output_text.delete("1.0", tk.END)
            # get the count for each sentiment
            label_counts = Counter([res["label"] for res in results])
            total = len(results)

            # display summary
            output_text.insert(tk.END, "===== Summary =====\n")
            output_text.insert(tk.END, "sentiment count (percentage)\n")
            output_text.insert(tk.END, "----------------------------\n")
            for label, count in label_counts.items():
               percentage = (count / total) * 100
               output_text.insert(tk.END, f"{label_to_sentiment[label]}: {count} ({percentage:.2f}%)\n")    
        except Exception as e:
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, f"Error processing file: {e}")

# Tkinter GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("600x500")

tk.Label(root, text="Enter Text:").pack()
text_entry = scrolledtext.ScrolledText(root, height=5, width=60)
text_entry.pack()

analyze_button = tk.Button(root, text="Analyze Text", command=analyze_text)
analyze_button.pack()

output_label = tk.Label(root, text="", fg="blue")
output_label.pack()

tk.Label(root, text="OR").pack()

file_button = tk.Button(root, text="Choose File", command=analyze_file)
file_button.pack()

output_text = scrolledtext.ScrolledText(root, height=10, width=70)
output_text.pack()

root.mainloop()
