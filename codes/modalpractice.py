import matplotlib.pyplot as plt

# Defining the fit_model function
def fit_model():
    # Mock-up example of training process
    history = {
        'accuracy': [0.6, 0.75, 0.85, 0.9],
        'val_accuracy': [0.55, 0.7, 0.8, 0.85],
        'loss': [0.7, 0.5, 0.35, 0.25],
        'val_loss': [0.75, 0.55, 0.4, 0.3]
    }
    model = None  # Replace with your actual model
    return history, model

# Running the model and training history
history, model = fit_model()

# Plotting training and validation accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
