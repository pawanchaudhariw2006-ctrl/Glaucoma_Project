# %%
# ==========================================
# STEP 1: IMPORTS & SETUP
# ==========================================
import os
import random
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
   
# Set Random Seeds for Reproducibility (Research Standard)
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("✅ Step 1: Libraries Loaded & Seeds Set.")

# %%
# ==========================================
# STEP 2: CONFIGURATION & DATA DISTRIBUTION
# ==========================================
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
MODEL_NAME = 'glaucoma_model.h5'

# Create a graph showing how many images are in each folder
def plot_data_distribution():
    categories = ['Glaucoma', 'Normal']
    counts = []
    for cat in categories:
        path = os.path.join(DATASET_PATH, cat)
        if os.path.exists(path):
            counts.append(len(os.listdir(path)))
        else:
            print(f"⚠️ Warning: Folder {path} not found.")
            counts.append(0)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=categories, y=counts, palette=['#d32f2f', '#388e3c'])
    plt.title('Distribution of Diseases in Dataset', fontsize=15)
    plt.ylabel('Number of Images')
    # Add numbers on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.savefig('research_1_distribution.png')
    print("✅ Step 2: Distribution Graph Saved as 'research_1_distribution.png'")

plot_data_distribution()

# %%
# ==========================================
# STEP 3: DATA LOADING & AUGMENTATION
# ==========================================
print("\nLoading Data Generators...")

# Advanced Augmentation to reach 95% accuracy
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False # Important for correct evaluation metrics
)
print(f"✅ Step 3: Data Loaded. Classes: {train_generator.class_indices}")

# %%
# ==========================================
# STEP 4: BUILD DENSENET121 MODEL
# ==========================================
print("\nBuilding Model...")

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Define Metrics to Track for Graphs
METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]

print("✅ Step 4: Model Architecture Built.")

# %%
# ==========================================
# STEP 5: STAGE 1 TRAINING (WARMUP)
# ==========================================
print("\n--- STARTING STAGE 1: WARMUP (10 Epochs) ---")

# Freeze the base model, only train the top layers
base_model.trainable = False 
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=METRICS)

history_warmup = model.fit(train_generator, epochs=10, validation_data=val_generator)
print("✅ Step 5: Warmup Complete.")

# %%
# ==========================================
# STEP 6: STAGE 2 TRAINING (FINE TUNING)
# ==========================================
print("\n--- STARTING STAGE 2: FULL TRAINING (50 Epochs) ---")

# Unfreeze everything for deep learning
base_model.trainable = True 

# Lower learning rate for delicate training
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=METRICS)

checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# Train for more epochs
history = model.fit(
    train_generator,
    epochs=50, 
    validation_data=val_generator,
    callbacks=[checkpoint, reduce_lr]
)
print("✅ Step 6: Full Training Complete. Best Model Saved.")

# %%
# ==========================================
# STEP 7: VISUALIZE TRAINING HISTORY (GRID)
# ==========================================
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot Accuracy
    axes[0,0].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[0,0].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2, linestyle='--')
    axes[0,0].set_title('Accuracy Over Epochs')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot Loss
    axes[0,1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0,1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[0,1].set_title('Loss Over Epochs')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot AUC
    axes[1,0].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[1,0].plot(history.history['val_auc'], label='Val AUC', linewidth=2, linestyle='--')
    axes[1,0].set_title('AUC Over Epochs')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot Recall (Sensitivity)
    axes[1,1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1,1].plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    axes[1,1].set_title('Recall Over Epochs')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('research_2_training_grid.png')
    print("✅ Step 7: History Grid Saved as 'research_2_training_grid.png'")

plot_training_history(history)

# %%
# ==========================================
# STEP 8: FINAL EVALUATION & CURVES
# ==========================================
print("\nGenerating Final Research Metrics...")

# Load best model for evaluation
best_model = load_model(MODEL_NAME)
val_generator.reset()

# Get Predictions
Y_pred_probs = best_model.predict(val_generator, verbose=1)
y_pred = (Y_pred_probs > 0.5).astype(int)
y_true = val_generator.classes

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Glaucoma', 'Normal'], 
            yticklabels=['Glaucoma', 'Normal'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('research_3_confusion_matrix.png')
print("✅ Saved Confusion Matrix")

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, Y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('research_4_roc_curve.png')
print("✅ Saved ROC Curve")

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, Y_pred_probs)
avg_precision = average_precision_score(y_true, Y_pred_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig('research_5_pr_curve.png')
print("✅ Saved Precision-Recall Curve")

# 4. Text Report
print("\n" + "="*50)
print(f"FINAL TEST ACCURACY: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("="*50)
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=['Glaucoma', 'Normal']))
print("="*50)
# %%
