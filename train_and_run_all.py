# -----------------------------
# 0. Imports
# -----------------------------
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# -----------------------------
# 1. Train Keras CNN
# -----------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model_keras = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_keras.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training Keras CNN...")
model_keras.fit(x_train, y_train, epochs=3, batch_size=64, verbose=2)
model_keras.save("mnist_cnn_keras.h5")
print("✔ Keras model saved.")

# -----------------------------
# 2. Train PyTorch CNN
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.fc1 = nn.Linear(64*7*7,64)
        self.fc2 = nn.Linear(64,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1,64*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_torch = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.001)

x_train_torch = torch.tensor(x_train, dtype=torch.float32).reshape(-1,1,28,28)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
x_test_torch = torch.tensor(x_test, dtype=torch.float32).reshape(-1,1,28,28)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

print("Training PyTorch CNN...")
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model_torch(x_train_torch.to(device))
    loss = criterion(outputs, y_train_torch.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/3], Loss: {loss.item():.4f}")

torch.save(model_torch.state_dict(), "mnist_cnn_pytorch.pt")
print("✔ PyTorch model saved.")

# -----------------------------
# 3. Train low-level TensorFlow network
# -----------------------------
x_train_flat = x_train.reshape(-1,28*28).astype(np.float32)
x_test_flat = x_test.reshape(-1,28*28).astype(np.float32)

n_input = 28*28
n_hidden1 = 128
n_hidden2 = 64
n_output = 10

W1 = tf.Variable(tf.random.normal([n_input,n_hidden1], stddev=0.1))
b1 = tf.Variable(tf.zeros([n_hidden1]))
W2 = tf.Variable(tf.random.normal([n_hidden1,n_hidden2], stddev=0.1))
b2 = tf.Variable(tf.zeros([n_hidden2]))
W3 = tf.Variable(tf.random.normal([n_hidden2,n_output], stddev=0.1))
b3 = tf.Variable(tf.zeros([n_output]))

optimizer_tf = tf.optimizers.Adam(0.001)
y_train_onehot = tf.one_hot(y_train, 10)

print("Training low-level TensorFlow network...")
for epoch in range(3):
    with tf.GradientTape() as tape:
        hidden1 = tf.nn.relu(tf.matmul(x_train_flat,W1)+b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1,W2)+b2)
        logits = tf.matmul(hidden2,W3)+b3
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train_onehot, logits=logits))
    grads = tape.gradient(loss, [W1,b1,W2,b2,W3,b3])
    optimizer_tf.apply_gradients(zip(grads,[W1,b1,W2,b2,W3,b3]))
    print(f"Epoch [{epoch+1}/3], Loss: {loss.numpy():.4f}")

np.save("W1.npy", W1.numpy())
np.save("b1.npy", b1.numpy())
np.save("W2.npy", W2.numpy())
np.save("b2.npy", b2.numpy())
np.save("W3.npy", W3.numpy())
np.save("b3.npy", b3.numpy())
print("✔ Low-level TensorFlow weights saved.")

# -----------------------------
# 4. GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a digit (0-9)")
        self.canvas = tk.Canvas(self,width=280,height=280,bg="white")
        self.canvas.pack()
        self.button = tk.Button(self,text="Predict",command=self.predict)
        self.button.pack()
        self.clear_btn = tk.Button(self,text="Clear",command=self.clear)
        self.clear_btn.pack()
        self.image = Image.new("L",(280,280),"white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>",self.paint)
    
    def paint(self,event):
        x1,y1 = (event.x-8),(event.y-8)
        x2,y2 = (event.x+8),(event.y+8)
        self.canvas.create_oval(x1,y1,x2,y2,fill="black")
        self.draw.ellipse([x1,y1,x2,y2],fill="black")
    
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,280,280],fill="white")
    
    def predict(self):
        img = self.image.resize((28,28))
        img = ImageOps.invert(img)
        img_array = np.array(img)/255.0
        img_tf = img_array.reshape(1,28,28,1).astype(np.float32)
        img_flat = img_array.reshape(1,28*28).astype(np.float32)
        img_torch = torch.tensor(img_array,dtype=torch.float32).reshape(1,1,28,28).to(device)
        result_text = ""
        # Keras
        pred_keras = np.argmax(model_keras.predict(img_tf))
        result_text += f"Keras CNN: {pred_keras}\n"
        # PyTorch
        with torch.no_grad():
            outputs = model_torch(img_torch)
            _, pred_torch = torch.max(outputs,1)
            result_text += f"PyTorch CNN: {pred_torch.item()}\n"
        # Low-level TF
        hidden1 = tf.nn.relu(tf.matmul(img_flat,W1)+b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1,W2)+b2)
        logits = tf.matmul(hidden2,W3)+b3
        pred_tf_raw = tf.argmax(logits, axis=1).numpy()[0]
        result_text += f"Low-level TF: {pred_tf_raw}\n"
        tk.messagebox.showinfo("Prediction", result_text)

if __name__=="__main__":
    import tkinter.messagebox
    app = App()
    app.mainloop()
