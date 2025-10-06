# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# import matplotlib.pyplot as plt

# # Đường dẫn đến các folders
# REAL_FREQ_PATH = '/Applications/Tien/deepfake/Dataset/features/real/frequency'
# REAL_MOBILE_PATH = '/Applications/Tien/deepfake/Dataset/features/real/mobilenet'
# FAKE_FREQ_PATH = '/Applications/Tien/deepfake/Dataset/features/fake/frequency'
# FAKE_MOBILE_PATH = '/Applications/Tien/deepfake/Dataset/features/fake/mobilenet'

# def load_features(freq_path, mobile_path, label):
#     """Load và kết hợp features từ frequency và mobilenet"""
#     features = []
#     labels = []
    
#     # Lấy danh sách files
#     freq_files = sorted([f for f in os.listdir(freq_path) if f.endswith('.npy')])
#     mobile_files = sorted([f for f in os.listdir(mobile_path) if f.endswith('.npy')])
    
#     # Đảm bảo số lượng files khớp nhau
#     common_files = min(len(freq_files), len(mobile_files))
    
#     print(f"Loading {common_files} samples for label {label}...")
    
#     for i in range(common_files):
#         try:
#             # Load frequency features
#             freq_feat = np.load(os.path.join(freq_path, freq_files[i]))
#             # Load mobilenet features
#             mobile_feat = np.load(os.path.join(mobile_path, mobile_files[i]))
            
#             # Flatten nếu cần
#             freq_feat = freq_feat.flatten()
#             mobile_feat = mobile_feat.flatten()
            
#             # Concatenate features
#             combined = np.concatenate([freq_feat, mobile_feat])
            
#             features.append(combined)
#             labels.append(label)
#         except Exception as e:
#             print(f"Error loading file {i}: {e}")
#             continue
    
#     return np.array(features), np.array(labels)

# # Load data
# print("Loading REAL samples...")
# X_real, y_real = load_features(REAL_FREQ_PATH, REAL_MOBILE_PATH, 0)

# print("Loading FAKE samples...")
# X_fake, y_fake = load_features(FAKE_FREQ_PATH, FAKE_MOBILE_PATH, 1)

# # Kết hợp data
# X = np.vstack([X_real, X_fake])
# y = np.concatenate([y_real, y_fake])

# print(f"\nTotal samples: {len(X)}")
# print(f"Feature dimension: {X.shape[1]}")
# print(f"Real samples: {np.sum(y == 0)}, Fake samples: {np.sum(y == 1)}")

# # Split data
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
# )

# # Normalize data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# # Reshape cho CNN (thêm channel dimension)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# print(f"\nTraining set: {X_train.shape}")
# print(f"Validation set: {X_val.shape}")
# print(f"Test set: {X_test.shape}")

# # Build CNN Model
# def create_cnn_model(input_shape):
#     model = models.Sequential([
#         # Block 1
#         layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
#         layers.BatchNormalization(),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.3),
        
#         # Block 2
#         layers.Conv1D(128, kernel_size=3, activation='relu'),
#         layers.BatchNormalization(),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.3),
        
#         # Block 3
#         layers.Conv1D(256, kernel_size=3, activation='relu'),
#         layers.BatchNormalization(),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.4),
        
#         # Block 4
#         layers.Conv1D(512, kernel_size=3, activation='relu'),
#         layers.BatchNormalization(),
#         layers.GlobalAveragePooling1D(),
#         layers.Dropout(0.5),
        
#         # Dense layers
#         layers.Dense(256, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
        
#         layers.Dense(128, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.4),
        
#         # Output layer
#         layers.Dense(1, activation='sigmoid')
#     ])
    
#     return model

# # Create model
# model = create_cnn_model((X_train.shape[1], 1))
# model.summary()

# # Compile model
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     loss='binary_crossentropy',
#     metrics=['accuracy', 
#              keras.metrics.Precision(name='precision'),
#              keras.metrics.Recall(name='recall'),
#              keras.metrics.AUC(name='auc')]
# )

# # Callbacks
# callbacks = [
#     EarlyStopping(
#         monitor='val_loss',
#         patience=15,
#         restore_best_weights=True,
#         verbose=1
#     ),
#     ModelCheckpoint(
#         'best_deepfake_model.h5',
#         monitor='val_accuracy',
#         save_best_only=True,
#         verbose=1
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-7,
#         verbose=1
#     )
# ]

# # Train model
# print("\nTraining model...")
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=100,
#     batch_size=32,
#     callbacks=callbacks,
#     verbose=1
# )

# # Evaluate on test set
# print("\nEvaluating on test set...")
# test_results = model.evaluate(X_test, y_test, verbose=1)
# print(f"\nTest Loss: {test_results[0]:.4f}")
# print(f"Test Accuracy: {test_results[1]:.4f}")
# print(f"Test Precision: {test_results[2]:.4f}")
# print(f"Test Recall: {test_results[3]:.4f}")
# print(f"Test AUC: {test_results[4]:.4f}")

# # Plot training history
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 3, 2)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 3, 3)
# plt.plot(history.history['auc'], label='Train AUC')
# plt.plot(history.history['val_auc'], label='Val AUC')
# plt.title('Model AUC')
# plt.xlabel('Epoch')
# plt.ylabel('AUC')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Predictions và confusion matrix
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns

# y_pred_proba = model.predict(X_test)
# y_pred = (y_pred_proba > 0.5).astype(int)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=['Real', 'Fake'],
#             yticklabels=['Real', 'Fake'])
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Save model
# model.save('deepfake_cnn_model.h5')
# print("\nModel saved as 'deepfake_cnn_model.h5'")

# # Save scaler
# import pickle
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
# print("Scaler saved as 'scaler.pkl'")