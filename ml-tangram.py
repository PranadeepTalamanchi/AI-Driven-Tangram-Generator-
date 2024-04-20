
import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageTk


# Load the Tangram dataset
def load_tangram_data(data_dir):
    tangram_data = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            word = folder_name.lower()
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".jpg"):
                    tangram_data.append({"Word": word, "Tangram": os.path.splitext(file_name)[0],
                                         "Image_Path": os.path.join(folder_path, file_name)})
    return pd.DataFrame(tangram_data)


# Preprocess the data
tangram_data = load_tangram_data("tangram/")

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tfidf.fit_transform(tangram_data['Word'])

# Train the model
knn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
knn_model.fit(tfidf_matrix)


def get_related_tangram(word):
    # Transform input word using the trained TF-IDF vectorizer
    word_tfidf = tfidf.transform([word])

    # Find the nearest tangram using the trained Nearest Neighbors model
    neighbor_index = knn_model.kneighbors(word_tfidf, return_distance=False)
    related_tangram = tangram_data.iloc[neighbor_index[0][0]]

    return related_tangram


def on_click():
    word = entry.get().lower()
    if word:
        related_tangram = get_related_tangram(word)
        render_image(related_tangram)
    else:
        messagebox.showwarning("Warning", "Please enter a word.")


def render_image(tangram):
    load = Image.open(tangram["Image_Path"])
    render = ImageTk.PhotoImage(load)
    img.configure(image=render)
    img.image = render


def create_ui():
    root = tk.Tk()
    root.title("Tangram Recommender")

    # Load and display an image
    default_image = tangram_data.iloc[0]  # Using the first image in the dataset as default
    load = Image.open(default_image["Image_Path"])
    render = ImageTk.PhotoImage(load)
    global img
    img = tk.Label(root, image=render)
    img.image = render
    img.pack()

    # Input box
    global entry
    entry = tk.Entry(root, width=40)
    entry.pack(pady=10)

    # Button
    button = tk.Button(root, text="Get Tangram", command=on_click)
    button.pack(pady=5)

    root.mainloop()


# Start UI
create_ui()
