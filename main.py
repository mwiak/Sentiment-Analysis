import tkinter as tk
from tkinter import filedialog, scrolledtext
from transformers import pipeline
import pandas as pd
import json
from collections import Counter

"""
main module that runs the GUI and perofrm the sentiment analysis on file or text
"""

# load sentiment analysis model pipeline from Hugging Face
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


# the model by default labels the pridictions LABEL_0 for negative, LABEL_1 for neutral and LABEL_2 for positive
label_to_sentiment = {
     "LABEL_0": 'negative',
     "LABEL_1": 'neutral',
     "LABEL_2": 'positive'
     }


def analyze_text():
    """
    performs the sentiment analysis on manually entered text and return the model response on the GUI
    """
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        result = classifier(text)[0]
        output_label.config(text=f"Sentiment: {label_to_sentiment[result['label']]} (Score: {result['score']:.2f})")

def analyze_file():
    """
    performs the sentiment analysis from selected JSON or CSV file and returns summary on the GUI
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
            results = classifier(texts)
            # delete the prevoius summary if exists
            output_text.delete("1.0", tk.END)
            # get the count for each sentiment
            label_counts = Counter([res["label"] for res in results])
            total = len(results)

            # Display summary
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
