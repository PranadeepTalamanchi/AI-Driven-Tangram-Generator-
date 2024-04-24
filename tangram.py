#This is sample demo code for few word inputs, you can use this code before using whole project in your local system.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

# Define Tangram shapes
shapes = {
    "cat": np.array([[0, 0], [1, 0], [1, 1], [0.5, 1.5], [0, 1], [0, 0]]),
    "dog": np.array([[0, 0], [1, 0], [1.5, 0.5], [1, 1], [0, 1], [0, 0]]),
    "bird": np.array([[0, 0], [1, 0], [1, 1], [0.5, 1.5], [0, 1], [0, 0], [0.5, 1.5], [0, 1], [-0.5, 1.5], [0, 0]]),
    "fish": np.array([[0, 0], [1, 0], [1.2, 0.2], [1, 0.4], [1, 0.6], [1.2, 0.8], [1, 1], [0, 1], [0, 0]]),
    "rabbit": np.array([[0, 0], [1, 0], [1.2, 0.2], [1.2, 0.8], [1, 1], [0, 1], [-0.2, 0.8], [-0.2, 0.2], [0, 0]]),
    "boat": np.array([[0, 0], [1, 0], [1.2, 0.2], [1, 0.4], [1, 0.6], [1.2, 0.8], [1, 1], [0, 1], [0, 0]]),
}

def plot_tangram(word):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')

    word = word.lower()

    if word in shapes:
        shape = shapes[word]
        color = "blue"
        plot_shape(shape, color, ax)
    else:
        st.error(f"Sorry, no Tangram shape found for '{word}'. Try another word.")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    plt.tight_layout()

    st.pyplot(fig)


def plot_shape(shape, color, ax):
    for i in range(len(shape) - 1):
        ax.plot([shape[i][0], shape[i + 1][0]],
                [shape[i][1], shape[i + 1][1]],
                color=color,
                lw=2)
    ax.fill(shape[:, 0], shape[:, 1], color=color, alpha=0.5)


def main():
    st.title("AI-Driven Tangram Generator 🎨")

    word = st.text_input("Enter a word to generate a Tangram:", "cat")
    st.write("Example words: cat, dog, bird, fish, rabbit, boat")

    if st.button("Generate Tangram"):
        plot_tangram(word)


if __name__ == "__main__":
    main()
