import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return image

def preprocess_image(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 1.4)
    return blurred_image

def detect_edges(blurred_image, threshold1, threshold2):
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    return edges

def clean_edges(edges):
    kernel = np.ones((2, 2), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return cleaned_edges

def invert_colors(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def process_image(image_path, threshold1=50, threshold2=50):
    image = load_image(image_path)
    blurred_image = preprocess_image(image)
    
    edges = detect_edges(blurred_image, threshold1, threshold2)
    cleaned_edges = clean_edges(edges)
    
    inverted_edges = invert_colors(cleaned_edges)
    
    final_edges = detect_edges(inverted_edges, threshold1, threshold2)
    cleaned_final_edges = clean_edges(final_edges)
    
    return image, inverted_edges

def display_images(original, processed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title('Processed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original, processed = process_image(file_path, threshold1=48, threshold2=53)
        display_images(original, processed)

def create_ui():
    root = Tk()
    root.title("Image Edge Detection")

    label = Label(root, text="Select an image to process and display")
    label.pack(pady=20)

    button = Button(root, text="Load Image", command=select_image)
    button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_ui()
