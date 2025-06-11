# STEP 1: Unzip Dataset
# STEP 1: Upload again if needed
from google.colab import files
uploaded = files.upload()  # Select your Dataset.zip here

# STEP 2: Unzip (change file name if needed)
!unzip -q "/content/Dataset.zip" -d /content/Dataset

# STEP 3: Updated Imports
import os
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# STEP 4 onward: rest remains same...

# STEP 3: Dataset Paths (Update these if folder names are different)
train_dir = '/content/Dataset/Train'
val_dir = '/content/Dataset/Validation'
model_path = 'xception_deepfake_model.keras'

# STEP 4: Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

# STEP 5: Model Architecture
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# STEP 6: Checkpoint to save model
checkpoint = ModelCheckpoint(model_path, save_best_only=False)

# STEP 7: Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    callbacks=[checkpoint],
    verbose=1
)

# STEP 8: Download trained model
from google.colab import files
files.download(model_path)
