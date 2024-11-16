# 📸 Meme Classifier

This project is a simple image classification model that distinguishes between two categories of memes: "Rickroll" and "Women and Cat Yelling". The model uses a convolutional neural network (CNN) architecture built with TensorFlow/Keras to train and classify meme images.

---


## ✨ Features

- 🖼️ Image Preprocessing: Loads and preprocesses images for model training.
- 🏋️‍♂️ CNN Model: A convolutional neural network is used for classifying meme images.
- 📊 Model Training: The model is trained using a dataset with two classes of images: "Rickroll" and "Women and Cat Yelling".
- 🤖 Image Prediction: The trained model is used to classify new meme images based on the learned features.

---



## ⚙️ Requirements

Before running this project, make sure to install the following libraries:

- `tensorflow` - For building and training the neural network.
- `matplotlib` - For displaying images.
- `opencv-python` - For image processing.
- `numpy` - For numerical operations.
- `os` - For file system interaction.

You can install the required libraries using pip:

```bash
pip install tensorflow matplotlib opencv-python numpy

```

# 🖼️ Model Architecture

The model uses the following layers:

1. **Conv2D (16 filters, 3x3 kernel)**: Extracts features from input images.
2. **MaxPool2D**: Reduces the dimensionality of the feature maps.
3. **Conv2D (32 filters, 3x3 kernel)**: Extracts more complex features.
4. **MaxPool2D**: Further reduces the dimensionality.
5. **Conv2D (64 filters, 3x3 kernel)**: Further feature extraction.
6. **MaxPool2D**: Reduces dimensionality again.
7. **Flatten**: Converts the 2D matrix to a 1D vector.
8. **Dense (512 neurons)**: Fully connected layer with ReLU activation.
9. **Dense (1 neuron)**: Output layer with a sigmoid activation function for binary classification.

---

# 📋 Usage

### Prepare the Dataset

Place meme images in the appropriate directories (`Training`, `Validation`, and `Testing`).

### Run the Model

Train the model using the training data with the following code:

```python
model.fit(train_dataset, steps_per_epoch=3, epochs=10, validation_data=validation_dataset)

```

## Classify New Images

After training, you can classify new images by loading them and predicting using the trained model:

```python
val = model.predict(images)
if val == 0:
    print("Rickroll meme")
else:
    print("Women and cat yelling meme")

```

# 📂 File Structure

```bash
/Meme Classifier/
├── Basedata/
│   ├── Training/
│   │   ├── Rickroll/
│   │   └── Women_and_Cat_Yelling/
│   ├── Validation/
│   │   ├── Rickroll/
│   │   └── Women_and_Cat_Yelling/
│   └── Testing/
│       ├── Rickroll/
│       └── Women_and_Cat_Yelling/
├── meme_classifier.py            # Main script for training and classifying memes
└── README.md                    # Project documentation

```


# 🔧 Conclusion

This Meme Classifier project uses deep learning techniques to classify meme images into two categories. By leveraging convolutional neural networks with TensorFlow/Keras, it achieves binary classification, which can be expanded for more complex tasks or additional categories.


