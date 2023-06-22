import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "../archive/"
# Load the saved model
model = tf.keras.models.load_model("./Saif_DL_Model/saved_model.pb")

# Load the test images
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(directory=path + 'test',
                                                  target_size=(224, 224),
                                                  color_mode="grayscale",
                                                  batch_size=1,
                                                  class_mode=None,
                                                  shuffle=False,
                                                  seed=42)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(directory=path +'train', 
                                                    target_size=(224, 224),
                                                    color_mode="grayscale",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True,seed=42)

# Evaluate the model
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()
model.evaluate(test_generator, steps=STEP_SIZE_TEST)

# Predict the output
test_generator.reset()
pred = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

# Get the class labels
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# Calculate accuracy
current_idx = 0
count_accurate = 0
Actual = []
for i in predictions:
    string = test_generator.filenames[current_idx]
    substr = '/'
    actual = string[:string.find(substr)]
    Actual.append(actual)
    pred = predictions[current_idx]
    if actual == pred:
        count_accurate += 1
    current_idx += 1
acc = count_accurate / test_generator.samples
print(f"The accuracy on predicting the test images is {round(acc*100, 2)}%.")

# Print 10 images and their predictions
current = [1, 37, 103, 189, 203, 274, 333, 355, 435, 478]
for i in current:
    plt.imshow(plt.imread(path + 'test/' + test_generator.filenames[i]))
    string = test_generator.filenames[i]
    substr = '/'
    actual = string[:string.find(substr)]
    plt.title(f"True: {actual}\nPredicted: {predictions[i]}")
    plt.show()

# Generate classification report
report = classification_report(Actual, predictions)
print(report)

# Save results to a CSV file
filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv("results.csv", index=False)
