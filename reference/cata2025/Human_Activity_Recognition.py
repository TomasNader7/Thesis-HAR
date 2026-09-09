# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Preprocessing and pipeline utilities
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model evaluation and selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Machine learning models
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier

# Deep learning with TensorFlow/Keras
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense
GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
to_categorical = tf.keras.utils.to_categorical

# Load the dataset using the file paths
X_train_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\X_train.txt"
y_train_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\y_train.txt"
X_test_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\X_test.txt"
y_test_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\y_test.txt"
activity_labels_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\activity_labels.txt"

# Load the datasets
X_train = pd.read_csv(X_train_path, header=None, delimiter=r'\s+')
y_train = pd.read_csv(y_train_path, header=None, delimiter=r'\s+').values.ravel()
X_test = pd.read_csv(X_test_path, header=None, delimiter=r'\s+')
y_test = pd.read_csv(y_test_path, header=None, delimiter=r'\s+').values.ravel()

# Load activity labels
activity_labels = pd.read_csv(activity_labels_path, header=None, delimiter=' ', index_col=0, names=['Activity'])
activity_names = dict(zip(activity_labels.index, activity_labels['Activity']))

# Map activity labels to the y data
y_train = pd.Series(y_train).map(activity_names)
y_test = pd.Series(y_test).map(activity_names)

# Convert categorical labels to integer indices
activity_name_to_index = {name: index for index, name in enumerate(y_train.unique())}
y_train_int = pd.Series(y_train).map(activity_name_to_index)
y_test_int = pd.Series(y_test).map(activity_name_to_index)

# Preprocessing pipeline for X_train and X_test
numeric_features = X_train.select_dtypes(include=['float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Define preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to X_train and X_test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Ensure alignment between X_train_preprocessed and y_train_int
print(f"Before alignment: X_train_preprocessed shape: {X_train_preprocessed.shape}, y_train_int shape: {y_train_int.shape}")
y_train_int = y_train_int[:X_train_preprocessed.shape[0]]  # Align if necessary
print(f"After alignment: X_train_preprocessed shape: {X_train_preprocessed.shape}, y_train_int shape: {y_train_int.shape}")

# Split dataset into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_preprocessed, y_train_int, test_size=0.2, random_state=42)

# Stacking model definition with added classifiers
def get_advanced_stacking_model():
    level0 = [
        ('perceptron', Perceptron(max_iter=2000, tol=1e-3)),
        ('random_forest', RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('xgboost', XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=42)),
        ('gradient_boosting', GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42))
    ]
    level1 = LogisticRegression(solver='saga', max_iter=2000, tol=1e-4)
    
    # Stacking model with 3-fold cross-validation
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return model

# Train the stacking model
stacking_model = get_advanced_stacking_model()
stacking_model.fit(X_train_split, y_train_split)

# Evaluate the stacking model on test set
y_pred_test_stacking = stacking_model.predict(X_test_preprocessed)
stacking_test_accuracy = accuracy_score(y_test_int, y_pred_test_stacking)
print(f"Advanced Stacking Model Test Accuracy: {stacking_test_accuracy}")

# Calculate and print accuracy of each base learner
base_learners = stacking_model.named_estimators_
for name, model in base_learners.items():
    model.fit(X_train_split, y_train_split)  # Fit model on training data
    y_pred = model.predict(X_test_preprocessed)  # Predict on test data
    accuracy = accuracy_score(y_test_int, y_pred)
    print(f"{name} Test Accuracy: {accuracy}")

# Enhanced CNN architecture
def create_advanced_cnn_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_split.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(filters=256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        GlobalAveragePooling1D(),  
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(len(y_train_split.unique()), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_split), y=y_train_split)
class_weights_dict = dict(enumerate(class_weights))

# Prepare data for CNN
X_train_cnn = np.expand_dims(X_train_split, axis=2)
X_val_cnn = np.expand_dims(X_val_split, axis=2)
y_train_cnn = to_categorical(y_train_split)
y_val_cnn = to_categorical(y_val_split)

# CNN training with class weights
cnn_model = create_advanced_cnn_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cnn, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val_cnn, y_val_cnn), 
    class_weight=class_weights_dict,  # Apply class weights
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate CNN on the test set
X_test_cnn = np.expand_dims(X_test_preprocessed, axis=2)
y_test_cnn = to_categorical(y_test_int)
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f"Enhanced CNN Test Accuracy: {cnn_test_accuracy}")

# Comparison of test accuracies
print(f"Advanced Stacking Model Test Accuracy: {stacking_test_accuracy}")
print(f"Enhanced CNN Test Accuracy: {cnn_test_accuracy}")

# Visualization

# 1. Plot training and validation accuracy and loss
def plot_cnn_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_cnn_history(cnn_history)
