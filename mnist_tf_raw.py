import tensorflow as tf
import numpy as np

# -----------------------------
# 1. Load MNIST dataset
# -----------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to [0,1] and flatten
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Convert to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

# -----------------------------
# 2. Initialize weights
# -----------------------------
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1))

W1 = init_weights([28*28, 128])
b1 = tf.Variable(tf.zeros([128]))

W2 = init_weights([128, 64])
b2 = tf.Variable(tf.zeros([64]))

W3 = init_weights([64, 10])
b3 = tf.Variable(tf.zeros([10]))

# -----------------------------
# 3. Forward pass
# -----------------------------
def forward(x):
    x = tf.matmul(x, W1) + b1
    x = tf.nn.relu(x)
    x = tf.matmul(x, W2) + b2
    x = tf.nn.relu(x)
    x = tf.matmul(x, W3) + b3
    return x

# -----------------------------
# 4. Loss function
# -----------------------------
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# -----------------------------
# 5. Optimizer
# -----------------------------
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# -----------------------------
# 6. Training loop
# -----------------------------
epochs = 5
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = forward(x_batch)
            loss = compute_loss(logits, y_batch)
        grads = tape.gradient(loss, [W1,b1,W2,b2,W3,b3])
        optimizer.apply_gradients(zip(grads, [W1,b1,W2,b2,W3,b3]))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.4f}")

np.save("W1.npy", W1.numpy())
np.save("b1.npy", b1.numpy())
np.save("W2.npy", W2.numpy())
np.save("b2.npy", b2.numpy())
np.save("W3.npy", W3.numpy())
np.save("b3.npy", b3.numpy())

# -----------------------------
# 7. Evaluate
# -----------------------------
correct = 0
total = 0
for x_batch, y_batch in test_ds:
    logits = forward(x_batch)
    pred = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct += tf.reduce_sum(tf.cast(pred==y_batch, tf.int32)).numpy()
    total += x_batch.shape[0]

print(f"✔ Test Accuracy: {100*correct/total:.2f}%")
