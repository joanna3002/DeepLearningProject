# DeepLearningProject

This project demonstrates **handwritten digit recognition (MNIST dataset)** using **three different deep learning frameworks**:

1. **Keras/TensorFlow CNN**  
2. **PyTorch CNN**  
3. **Low-level TensorFlow fully connected network**

It also includes an **interactive GUI** to draw digits and predict using all three models.

---

## **Project Structure**

DeepLearningProject/
│
├─ mnist_cnn_keras.h5 # Trained Keras CNN model
├─ mnist_cnn_pytorch.pt # Trained PyTorch CNN model
├─ W1.npy, b1.npy, W2.npy, b2.npy, W3.npy, b3.npy # Low-level TF weights
├─ train_and_run_all.py # All-in-one training & GUI script
├─ draw_and_predict_all.py # GUI script (if used separately)
├─ mnist_tf_raw.py # Low-level TF training script
├─ mnist_cnn_pytorch.py # PyTorch training script
├─ mnist_cnn.py # Keras training script
└─ README.md # This file
---

## **Requirements**

- Python 3.12+  
- Packages:
  ```bash

   (Optional) Create a virtual environment:
  pip install tensorflow torch matplotlib pillow numpy
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt


How to Run:

Activate your virtual environment (if using):

.\venv\Scripts\activate


Run the all-in-one script:

python train_and_run_all.py


This trains all three models (or loads saved models if they exist)

Opens the GUI to draw digits and predict

Draw a digit (0-9) on the canvas → click Predict → see results from:

*Keras CNN

*PyTorch CNN

*Low-level TensorFlow network

*Click Clear to draw again.
