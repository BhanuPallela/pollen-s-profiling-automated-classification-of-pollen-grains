'''# test_model.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('model.h5')

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy:.4f}")
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load_model('model.h5')

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict and Evaluate
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_generator.classes) if test_generator.class_mode == 'categorical' else test_generator.classes

# Accuracy
test_acc = accuracy_score(test_generator.classes, y_pred)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
print("\n📊 Confusion Matrix:")
disp.plot(cmap=plt.cm.Blues)
plt.title("Test Data Confusion Matrix")
plt.show()