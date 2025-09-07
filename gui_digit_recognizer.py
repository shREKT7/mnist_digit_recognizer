# gui_digit_recognizer.py - Fixed with preprocessing + Resampling fix
import tkinter as tk
from PIL import Image, ImageOps, ImageFilter, ImageGrab
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "models/mnist_cnn.h5"

# ==============================
# Preprocessing function
# ==============================
def preprocess_image_for_mnist(pil_img):
    """
    Takes a PIL Image (from canvas) and returns a (1,28,28,1) numpy array normalized [0,1].
    """
    img = pil_img.convert('L')         # grayscale
    img = ImageOps.invert(img)         # make digit white, background black
    img = img.filter(ImageFilter.MaxFilter(3))

    arr = np.array(img)
    thresh = 50
    mask = arr > thresh
    if mask.sum() == 0:
        return None

    # Crop bounding box
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img.crop((x0, y0, x1, y1))

    # Resize proportionally
    max_side = max(cropped.size)
    scale = 20.0 / max_side
    new_w = int(cropped.size[0] * scale)
    new_h = int(cropped.size[1] * scale)

    # ✅ FIX: Use modern resampling
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center into 28x28
    new_img = Image.new('L', (28,28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    new_img.paste(resized, (left, top))

    arr_final = np.array(new_img).astype('float32') / 255.0
    arr_final = arr_final.reshape(1,28,28,1)
    return arr_final

# ==============================
# Tkinter App
# ==============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("350x400")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=280, height=280, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.label = tk.Label(self, text="Draw a digit and click Predict", font=("Helvetica", 12))
        self.label.grid(row=1, column=0, columnspan=4)

        self.predict_btn = tk.Button(self, text="Predict", command=self.classify_handwriting)
        self.predict_btn.grid(row=2, column=0, pady=10)

        self.clear_btn = tk.Button(self, text="Clear", command=self.clear)
        self.clear_btn.grid(row=2, column=1, pady=10)

        self.quit_btn = tk.Button(self, text="Quit", command=self.destroy)
        self.quit_btn.grid(row=2, column=2, pady=10)

        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
        else:
            self.label.configure(text="Model not found. Train first!")

    def draw(self, event):
        r = 8
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit and click Predict")

    def classify_handwriting(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))

        processed = preprocess_image_for_mnist(img)
        if processed is None:
            self.label.configure(text="Draw a digit first")
            return

        pred = self.model.predict(processed)
        digit = np.argmax(pred)
        prob = np.max(pred)
        self.label.configure(text=f"Prediction: {digit} ({prob*100:.2f}%)")

if __name__ == "__main__":
    app = App()
    app.mainloop()
