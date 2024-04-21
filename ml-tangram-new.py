
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
    load = load.resize((250, 250), Image.LANCZOS)  # Change ANTIALIAS to LANCZOS
    render = ImageTk.PhotoImage(load)
    img.configure(image=render)
    img.image = render


def create_ui():
    root = tk.Tk()
    root.title("Tangram Recommender")
    root.geometry("400x600")
    root.configure(bg="black")
    # # Background image
    # background_image = Image.open("C:/Users/prana/Desktop/zo.jpg")  # Replace "background_image.jpg" with your image path
    # background_image = background_image.resize((2060, 600), Image.LANCZOS)
    # photo = ImageTk.PhotoImage(background_image)
    # background_label = tk.Label(root, image=photo)
    # background_label.image = photo
    # background_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Title
    title_label = tk.Label(root, text="AI-Driven Tangram Generator 🎨", font=("Helvetica", 20, "bold"), fg="white", bg="black")
    title_label.place(relx=0.5, rely=0.05, anchor="center")

    # Input box
    global entry
    entry = tk.Entry(root, width=30, font=("Helvetica", 14))
    entry.insert(0, "cat")
    entry.place(relx=0.5, rely=0.15, anchor="center")

    # Example words
    example_words_label = tk.Label(root, text="Example words: horse, swan, chicken, cat1, house2, dog, fish, rabbit, boat", font=("Helvetica", 12), fg="black")
    example_words_label.place(relx=0.5, rely=0.2, anchor="center")

    # Button
    button = tk.Button(root, text="Generate Tangram", command=on_click, font=("Helvetica", 14), fg="black", bg="white")
    button.place(relx=0.5, rely=0.28, anchor="center")

    # Load and display an image
    default_image = tangram_data.iloc[0]  # Using the first image in the dataset as default
    load = Image.open(default_image["Image_Path"])
    load = load.resize((250, 250), Image.LANCZOS)  # Change ANTIALIAS to LANCZOS
    render = ImageTk.PhotoImage(load)
    global img
    img = tk.Label(root, image=render)
    img.image = render
    img.place(relx=0.5, rely=0.48, anchor="center")

    # # Generated Tangram
    # tangram_label = tk.Label(root, text="Generated Tangram", font=("Helvetica", 16, "bold"), fg="#333", bg="#ffc107")
    # tangram_label.place(relx=0.5, rely=0.78, anchor="center")
    #
    # # Display generated tangram
    # generated_tangram = tk.Label(root, font=("Helvetica", 16), bg="#ffc107", bd=2, relief="groove", width=20, height=2)
    # generated_tangram.place(relx=0.5, rely=0.85, anchor="center")

    def generate_tangram():
        word = entry.get().lower()
        if word:
            related_tangram = get_related_tangram(word)
            # generated_tangram.configure(text=related_tangram["Word"].capitalize())
            render_image(related_tangram)
        else:
            messagebox.showwarning("Warning", "Please enter a word.")

    # Button for generating tangram
    # generate_button = tk.Button(root, text="Generate Tangram", command=generate_tangram, font=("Helvetica", 14), bg="#2196f3", fg="white")
    # generate_button.place(relx=0.5, rely=0.93, anchor="center")

    root.mainloop()


# Start UI
create_ui()
