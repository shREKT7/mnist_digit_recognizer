# train.py - Improved CNN with Augmentation & Callbacks
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, callbacks, utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# ==============================
# 1) Load and preprocess dataset
# ==============================
(num_classes := 10)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

y_train_cat = utils.to_categorical(y_train, num_classes)
y_test_cat  = utils.to_categorical(y_test, num_classes)

# ==============================
# 2) Build stronger CNN
# ==============================
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn()
model.summary()

# ==============================
# 3) Data Augmentation
# ==============================
datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12
)
datagen.fit(x_train)

# ==============================
# 4) Callbacks
# ==============================
ckpt = callbacks.ModelCheckpoint("models/mnist_cnn.h5", save_best_only=True, monitor="val_loss")
es   = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
rlr  = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ==============================
# 5) Train model
# ==============================
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=128),
    validation_data=(x_test, y_test_cat),
    steps_per_epoch=len(x_train)//128,
    epochs=25,
    callbacks=[ckpt, es, rlr],
    verbose=2
)

# ==============================
# 6) Evaluate
# ==============================
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"✅ Test accuracy: {test_acc:.4f}")

# Classification Report & Confusion Matrix
y_pred = model.predict(x_test).argmax(axis=1)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ==============================
# 7) Training curves
# ==============================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.savefig("training_curves.png", dpi=150)
plt.show()
