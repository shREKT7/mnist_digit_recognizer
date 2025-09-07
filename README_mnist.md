# 🔢 Handwritten Digit Recognition with Tkinter GUI

This project is a **Python deep learning application** that classifies handwritten digits (0–9) using the **MNIST dataset**.  
It combines a **Convolutional Neural Network (CNN)** for digit recognition with a **Tkinter-based GUI** where users can draw digits and get instant predictions.

---

## 🚀 Features

- **Digit Classification**: Recognizes digits `0–9` from 28×28 grayscale images.  
- **Convolutional Neural Network (CNN)**: Trained on the MNIST dataset with >98% accuracy.  
- **Tkinter GUI**: Interactive window to draw digits and predict in real time.  
- **Data Augmentation**: Robust model trained with shifts, rotations, and zooms.  
- **Visualization**: Training curves and confusion matrix for evaluation.  
- **Model Persistence**: Trained model is saved (`mnist_cnn.h5`) and reused for predictions.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+  
- **Deep Learning Framework:** TensorFlow / Keras  
- **GUI Framework:** Tkinter  
- **Visualization:** Matplotlib, Seaborn  
- **Dataset:** MNIST (60,000 train / 10,000 test images)  

---

## 📂 Project Structure

```
mnist_digit_recognizer/
│
├── train.py                  # Train and evaluate CNN model
├── gui_digit_recognizer.py   # Tkinter GUI for digit drawing & prediction
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── models/                   # Saved trained models (.h5)
├── output/                   # Training curves & confusion matrix
└── assets/                   # Screenshots for demo
```

---

## ⚡ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mnist_digit_recognizer.git
   cd mnist_digit_recognizer
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### 🔹 Train the Model
```bash
python train.py
```

- Trains CNN on MNIST dataset with data augmentation.  
- Saves best model in `models/mnist_cnn.h5`.  
- Outputs training curves and confusion matrix in `output/`.  

### 🔹 Run the GUI
```bash
python gui_digit_recognizer.py
```

- Opens an interactive Tkinter app.  
- Draw digits on the canvas and click **Predict**.  
- Model predicts digit with confidence percentage.  

---

## 📸 Screenshots

| GUI Canvas | Prediction Example |
|------------|---------------------|
| ![canvas](assets/screenshot_gui.png) | ![prediction](assets/screenshot_prediction.png) |

---

## 🧩 How It Works (Under the Hood)

1. **Data Preprocessing**  
   - Normalize MNIST images to [0,1].  
   - Reshape into `(28,28,1)` for CNN input.  

2. **CNN Architecture**  
   - Multiple Conv2D + MaxPooling layers.  
   - Dense fully connected layer with Dropout for regularization.  
   - Softmax output (10 classes).  

3. **Training**  
   - Optimizer: Adam.  
   - Loss: Categorical Crossentropy.  
   - Metrics: Accuracy.  
   - Data augmentation improves robustness.  

4. **GUI Predictions**  
   - Canvas image cropped, centered, resized to 28×28.  
   - Converted to grayscale & normalized.  
   - Fed into CNN → predicted digit + confidence %.  

---

## 🔮 Future Enhancements

- Deploy as a **web app** (Flask/FastAPI).  
- Convert into a **mobile app** (Kivy/Flutter + TensorFlow Lite).  
- Integrate with **OCR pipelines** for real-world handwriting recognition.  
- Add **live probability bar chart** in GUI for all 10 digits.  

---

## 📜 License

MIT License © 2025 — Uzair Teli  
Free to use, modify, and distribute with attribution.

---

## 🤝 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)  
- [Tkinter GUI](https://docs.python.org/3/library/tkinter.html)
