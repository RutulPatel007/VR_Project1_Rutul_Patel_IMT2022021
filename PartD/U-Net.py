import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class MSFDDataset(tf.keras.utils.Sequence):
    def __init__(self, image_files, segmentation_files, img_size=(128, 128), batch_size=32):
        self.image_files = image_files
        self.segmentation_files = segmentation_files
        self.img_size = img_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.image_files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_segmentations = self.segmentation_files[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        images = []
        segmentations = []
        
        for img_path, seg_path in zip(batch_images, batch_segmentations):
            image = cv2.imread(img_path)
            if image is None:
                continue  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size[1], self.img_size[0])).astype(np.float32) / 255.0
            
            segmentation = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if segmentation is None:
                continue  
            segmentation = cv2.resize(segmentation, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            segmentation = segmentation.astype(np.uint8)  
            
            images.append(image)
            segmentations.append(segmentation)
        
        images = np.array(images, dtype=np.float32)
        segmentations = np.array(segmentations, dtype=np.uint8)
        
        return images, segmentations

# Load dataset paths
dataset_path = "/kaggle/working/MSFD/1"
face_dir = os.path.join(dataset_path, "face_crop")
segmentation_dir = os.path.join(dataset_path, "face_crop_segmentation")

image_files = []
segmentation_files = []

for img_name in os.listdir(face_dir):
    img_path = os.path.join(face_dir, img_name)
    seg_path = os.path.join(segmentation_dir, img_name)
    
    if os.path.exists(img_path) and os.path.exists(seg_path):
        image_files.append(img_path)
        segmentation_files.append(seg_path)

# Train-test split (80-20)
train_images, val_images, train_masks, val_masks = train_test_split(image_files, segmentation_files, test_size=0.2, random_state=42)

# Create dataset instances
train_dataset = MSFDDataset(train_images, train_masks)
val_dataset = MSFDDataset(val_images, val_masks)

print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")



import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Convert dataset to NumPy arrays
def get_data_from_generator(generator):
    X, Y = [], []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        X.append(x_batch)
        Y.append(y_batch)
    return np.concatenate(X), np.concatenate(Y)

# Load datasets
X_train, Y_train = get_data_from_generator(train_dataset)
X_val, Y_val = get_data_from_generator(val_dataset)

# Ensure masks are in the correct format
Y_train = Y_train[..., np.newaxis] / 255.0  # Normalize mask
Y_val = Y_val[..., np.newaxis] / 255.0

# IoU and Dice Metrics
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

# U-Net Model Definition
def build_unet():
    input_shape = (128, 128, 3)
    dropout_rate = 0.3
    use_batchnorm = True
    base_filters = 32
    depth = 3

    inputs = tf.keras.layers.Input(input_shape)
    x = inputs
    skips = []

    # Encoder
    for i in range(depth):
        filters = base_filters * (2**i)
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Bottleneck
    filters = base_filters * (2**depth)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    # Decoder
    for i in reversed(range(depth)):
        filters = base_filters * (2**i)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skips[i]])
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", iou_metric, dice_metric]
    )
    return model

# Build and train the model
model = build_unet()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=50,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]
)

# Save best model
model.save("unet_best.h5")
