import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_and_preprocess_data():
    """Loads MNIST dataset and applies normalization & reshaping."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    """Defines and returns a CNN model for MNIST classification."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=5):
    """Compiles and trains the CNN model."""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluates model on test dataset and returns accuracy."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.2f}")

def plot_training_history(history):
    """Plots training vs validation accuracy."""
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(model, x_test, y_test):
    """Displays confusion matrix."""
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

def display_sample_predictions(model, x_test, y_test, num_samples=3):
    """Displays sample predictions with ground truth labels."""
    y_pred = np.argmax(model.predict(x_test), axis=1)
    for i in range(num_samples):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[i]} | Predicted: {y_pred[i]}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    cnn_model = create_cnn_model()
    training_history = train_model(cnn_model, x_train, y_train, x_test, y_test)
    evaluate_model(cnn_model, x_test, y_test)
    plot_training_history(training_history)
    plot_confusion_matrix(cnn_model, x_test, y_test)
    display_sample_predictions(cnn_model, x_test, y_test)
  
