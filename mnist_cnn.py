import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN needs shape: (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# -----------------------------
# 2. Build CNN Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# -----------------------------
# 3. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 4. Train Model
# -----------------------------
model.fit(x_train, y_train, epochs=5, batch_size=64)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print(f"✔ Test Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 6. Show Predictions on Images
# -----------------------------
predictions = model.predict(x_test)

# Show 5 sample images with predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Prediction: {np.argmax(predictions[i])} | Actual: {y_test[i]}")
    plt.axis("off")
    plt.show()
