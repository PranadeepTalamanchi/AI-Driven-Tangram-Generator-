import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
import numpy as np
from skimage import transform as tf
from transformers import ViTFeatureExtractor, TFAutoModel
from io import BytesIO

class TangramGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Tangram Generator")

        # Label and Entry for the word
        self.label = tk.Label(master, text="Enter a Word:")
        self.label.pack()

        self.entry = tk.Entry(master)
        self.entry.pack()

        # Button to generate tangram
        self.generate_button = tk.Button(master, text="Generate Tangram", command=self.generate_tangram)
        self.generate_button.pack()

        # Display area for the generated image and its corresponding tangram
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()

    def generate_tangram(self):
        word = self.entry.get()
        if word.strip() == "":
            messagebox.showerror("Error", "Please enter a word.")
            return

        # Generate image and tangram
        generated_image = generate_image(word)
        generated_tangram = generate_tangram_from_image(generated_image)

        # Display the generated image and tangram
        self.display_image(generated_image, 50, 50)
        self.display_image(generated_tangram, 500, 50)

    def display_image(self, image, x, y):
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(x, y, anchor="nw", image=image)
        self.canvas.image = image

def generate_image(word):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # API call to Unsplash using the word entered by the user as the query parameter
    response = requests.get(f"https://source.unsplash.com/400x400/?{word}")
    image = Image.open(BytesIO(response.content))

    inputs = feature_extractor(images=image, return_tensors="tf")
    outputs = model(inputs.pixel_values)
    
    output_image = outputs.last_hidden_state[0].numpy()
    output_image = np.transpose(output_image, (1, 0, 2))  # Correct the axis for the transposition

    img = Image.fromarray((output_image * 255).astype(np.uint8))
    return img

def generate_tangram_from_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Convert to numpy array
    img_array = np.array(grayscale_image)

    # Thresholding
    img_array[img_array < 128] = 0
    img_array[img_array >= 128] = 255

    # Hough transform
    h, theta, d = tf.hough_line(img_array)

    # Create tangram from Hough lines
    tangram = Image.new("RGB", image.size, (255, 255, 255))

    # Tangram shapes
    shapes = []

    # Choose top 7 lines
    for _, angle, dist in zip(*tf.hough_line_peaks(h, theta, d, num_peaks=7)):
        angle = np.rad2deg(angle)
        dist = dist - img_array.shape[0] / 2
        shapes.append((angle, dist))

    for angle, dist in shapes:
        matrix = tf.EuclideanTransform(rotation=np.deg2rad(angle), translation=(0, dist))
        rotated_img = tf.rotate(img_array, angle, resize=True, center=(img_array.shape[1] / 2, img_array.shape[0] / 2))
        transformed_img = tf.warp(rotated_img, matrix.inverse)
        tangram.paste(Image.fromarray((transformed_img * 255).astype(np.uint8)), (0, 0))

    return tangram

root = tk.Tk()
app = TangramGeneratorApp(root)
root.mainloop()
