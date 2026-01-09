import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from glob import glob
import json

class SanskritCharRecognizer:
    def __init__(self, img_size=(128, 128), num_classes=50):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_to_char = {}
        self.char_to_class = {}
    
    def build_model(self):
        """Build a simple CNN for character recognition"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 1)),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_labeled_data(self, data_dir):
        """
        Load labeled data from directory structure:
        data_dir/
            class_name_1/
                image1.png
                image2.png
            class_name_2/
                ...
        """
        X = []
        y = []
        
        # Get all class folders
        class_folders = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
        
        if not class_folders:
            raise ValueError(f"No class folders found in {data_dir}")
        
        # Create class mappings
        class_folders.sort()
        self.class_to_char = {i: name for i, name in enumerate(class_folders)}
        self.char_to_class = {name: i for i, name in enumerate(class_folders)}
        
        print(f"Found {len(class_folders)} character classes")
        
        # Load images
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(data_dir, class_name)
            image_files = glob(os.path.join(class_path, '*.png'))
            
            print(f"  Loading {class_name}: {len(image_files)} images")
            
            for img_path in image_files:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize
                    X.append(img)
                    y.append(class_idx)
        
        X = np.array(X).reshape(-1, *self.img_size, 1)
        y = np.array(y)
        
        print(f"\nLoaded {len(X)} total images")
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=10, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, image):
        """Predict character from image"""
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, self.img_size)
        image = image / 255.0
        image = image.reshape(1, *self.img_size, 1)
        
        prediction = self.model.predict(image, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        
        return self.class_to_char[class_idx], confidence
    
    def save_model(self, model_path, mappings_path):
        """Save model and class mappings"""
        self.model.save(model_path)
        
        mappings = {
            'class_to_char': self.class_to_char,
            'char_to_class': self.char_to_class
        }
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Mappings saved to {mappings_path}")
    
    def load_model(self, model_path, mappings_path):
        """Load model and class mappings"""
        self.model = keras.models.load_model(model_path)
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        # Convert string keys back to integers for class_to_char
        self.class_to_char = {int(k): v for k, v in mappings['class_to_char'].items()}
        self.char_to_class = mappings['char_to_class']
        
        print(f"Model loaded from {model_path}")

# Training script
def train_character_recognizer():
    """Train the character recognition model"""
    
    # Paths
    train_dir = 'data/labeled/train'
    model_save_path = 'models/trained/char_recognizer.keras'
    mappings_save_path = 'models/trained/char_mappings.json'
    
    # Create output directory
    os.makedirs('models/trained', exist_ok=True)
    
    # Initialize model
    recognizer = SanskritCharRecognizer(img_size=(128, 128))
    
    # Load data
    print("Loading labeled data...")
    X, y = recognizer.load_labeled_data(train_dir)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Build and train model
    print("\nBuilding model...")
    recognizer.build_model()
    recognizer.model.summary()
    
    print("\nTraining model...")
    history = recognizer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Save model
    recognizer.save_model(model_save_path, mappings_save_path)
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_character_recognizer()