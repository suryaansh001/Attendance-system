import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the directory containing the image folders
image_folder = 'dataset1'

def preprocess_images(image_folder, image_size=(224, 224)):
    data = []
    labels = []
    for root, dirs, files in os.walk(image_folder):
        for dir_name in dirs:
            class_folder = os.path.join(root, dir_name)
            for file_name in os.listdir(class_folder):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_folder, file_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image at {img_path}")
                        continue
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize to [0, 1]
                    data.append(img)
                    labels.append(dir_name)  # Use the folder name as the label
    print(f"Total images processed: {len(data)}")
    return np.array(data), np.array(labels)

# Process images
data, labels = preprocess_images(image_folder)

# Shuffle the data
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Encoding the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load VGG16 model with pretrained weights, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(len(label_encoder.classes_), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for saving the best model and reducing learning rate on plateau
checkpoint = ModelCheckpoint('best_model_vgg.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[checkpoint, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Save the label encoder
with open('/home/surya/Desktop/proj/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Print model summary

print(model.summary())
#save model
model.save('kvupdatesmodel.keras')

