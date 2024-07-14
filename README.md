# COVID-19 Detection using CNN

This project aims to detect COVID-19 from chest X-ray images using a Convolutional Neural Network (CNN) implemented in Keras.

## Images
![image](https://github.com/user-attachments/assets/5ce2f9ce-72e8-4094-a341-c276e3497840) ![image](https://github.com/user-attachments/assets/cde2a218-4b8f-45fe-be59-cbb9c76a597f)



## Dataset

The dataset consists of chest X-ray images categorized into two classes:
- COVID
- Normal

The dataset is divided into training and validation sets:

```
CovidDataset/
    Train/
        Covid/
        Normal/
    Val/
        Covid/
        Normal/
```

## Model Architecture

The CNN model has the following architecture:

- 4 convolutional layers with ReLU activation and max-pooling
- Dropout layers to prevent overfitting
- A flattening layer
- A dense layer with ReLU activation
- A dropout layer
- An output layer with sigmoid activation for binary classification

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# Define paths
TRAIN_PATH = 'CovidDataset/Train/'
VAL_PATH = 'CovidDataset/Val/'

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(224,224,3)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["accuracy"])

# Model summary
model.summary()
```

## Data Preprocessing

```python
# Data augmentation for training set
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation set
test_dataset = image.ImageDataGenerator(rescale=1./255)

# Generate training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

# Generate validation data
validation_generator = test_dataset.flow_from_directory(
    VAL_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)
```

## Training the Model

```python
# Train the model
hist = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)
```

## Evaluation

### Accuracy and Loss Plots

```python
# Plot accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

### Testing the Model

```python
import numpy as np
import cv2
from keras.preprocessing import image

# Test with a COVID-19 positive case
xtest_image = image.load_img('CovidDataset/Val/Covid/16654_1_1.png', target_size=(224, 224))
xtest_image = image.img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis=0)
results = model.predict(xtest_image)
imggg = cv2.imread('CovidDataset/Val/Covid/16654_1_1.png')
imggg = np.array(imggg)
imggg = cv2.resize(imggg, (400, 400))
plt.imshow(imggg)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model: ", prediction)

# Test with a COVID-19 negative case
xtest_image = image.load_img('CovidDataset/Val/Normal/NORMAL2-IM-0395-0001.jpeg', target_size=(224, 224))
xtest_image = image.img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis=0)
results = model.predict(xtest_image)
imggg = cv2.imread('CovidDataset/Val/Normal/NORMAL2-IM-0395-0001.jpeg')
imggg = np.array(imggg)
imggg = cv2.resize(imggg, (400, 400))
plt.imshow(imggg)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model: ", prediction)
```

## Results

The model achieved a validation accuracy of 96.67% by the end of 10 epochs.
