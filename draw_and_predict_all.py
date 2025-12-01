import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

# -----------------------------
# 1. Load TensorFlow/Keras model
# -----------------------------
try:
    model_keras = tf.keras.models.load_model("mnist_cnn_keras.h5")
except:
    model_keras = None
    print("TensorFlow/Keras model not found.")

# -----------------------------
# 2. Load PyTorch model
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_torch = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model_torch.load_state_dict(torch.load("mnist_cnn_pytorch.pt", map_location=device))
    model_torch.eval()
    model_torch.to(device)
except:
    model_torch = None
    print("PyTorch model not found.")

# -----------------------------
# 3. Load low-level TensorFlow model weights
# -----------------------------
# Example: weights from mnist_tf_raw.py training
try:
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")
    W3 = np.load("W3.npy")
    b3 = np.load("b3.npy")
except:
    W1 = b1 = W2 = b2 = W3 = b3 = None
    print("Low-level TensorFlow weights not found.")

def tf_forward(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.nn.relu(tf.matmul(x, W1) + b1)
    x = tf.nn.relu(tf.matmul(x, W2) + b2)
    x = tf.matmul(x, W3) + b3
    return tf.argmax(x, axis=1).numpy()[0]

# -----------------------------
# 4. Tkinter GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a digit (0-9)")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.button = tk.Button(self, text="Predict", command=self.predict)
        self.button.pack()
        self.clear_btn = tk.Button(self, text="Clear", command=self.clear)
        self.clear_btn.pack()
        
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
    
    def paint(self, event):
        x1, y1 = (event.x-8), (event.y-8)
        x2, y2 = (event.x+8), (event.y+8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")
    
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,280,280], fill="white")
    
    def predict(self):
        img = self.image.resize((28,28))
        img = ImageOps.invert(img)
        img_array = np.array(img)/255.0
        img_tf = img_array.reshape(1,28,28,1).astype(np.float32)
        img_flat = img_array.reshape(1,28*28).astype(np.float32)
        img_torch = torch.tensor(img_array, dtype=torch.float32).reshape(1,1,28,28).to(device)
        
        result_text = ""
        
        # TensorFlow/Keras prediction
        if model_keras:
            pred_keras = np.argmax(model_keras.predict(img_tf))
            result_text += f"Keras CNN: {pred_keras}\n"
        
        # PyTorch prediction
        if model_torch:
            with torch.no_grad():
                outputs = model_torch(img_torch)
                _, pred_torch = torch.max(outputs,1)
                result_text += f"PyTorch CNN: {pred_torch.item()}\n"
        
        # Low-level TensorFlow prediction
        if W1 is not None:
            pred_tf_raw = tf_forward(img_flat)
            result_text += f"Low-level TF: {pred_tf_raw}\n"
        
        tk.messagebox.showinfo("Prediction", result_text)

# -----------------------------
# 5. Run the app
# -----------------------------
if __name__ == "__main__":
    import tkinter.messagebox
    app = App()
    app.mainloop()
