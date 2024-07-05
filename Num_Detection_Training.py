import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Ensure TensorFlow logging is suppressed (just for pycharm)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to the dataset and parameters for splitting
path = 'myData'
test_ratio = 0.2
validation_ratio = 0.2
image_dim = (32, 32, 1)  # Grayscale images, so channels = 1

# Lists to store images and their corresponding class labels
images = []
class_no = []

# Get the list of class directories
my_list = os.listdir(path)
no_of_classes = len(my_list)
print("No of Classes: ", no_of_classes)

# Importing images and creating labels
print("Importing Classes.....")
for i in range(no_of_classes):
    class_path = os.path.join(path, str(i))
    my_pic_list = os.listdir(class_path)
    for j in my_pic_list:
        image_path = os.path.join(class_path, j)
        cur_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if cur_image is not None:
            cur_image = cv2.resize(cur_image, (image_dim[0], image_dim[1]))
            images.append(cur_image)
            class_no.append(i)
        else:
            print(f"Failed to load image: {image_path}")
    print(i, end=" ")
print()

# Convert lists to numpy arrays
images = np.array(images)
class_no = np.array(class_no)
print(f"Total images loaded: {len(images)}\n{images.shape}")

# Splitting data into train, test, and validation sets
x_train, x_test, y_train, y_test = train_test_split(images, class_no, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

# Calculate number of samples per class in the training set
no_of_samples = []
for i in range(no_of_classes):
    no_of_samples.append(np.sum(y_train == i))

print(no_of_samples)

# Plotting the number of samples per class
plt.figure(figsize=(10, 5))
plt.bar(range(0, no_of_classes), no_of_samples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("No of Images")
plt.show()

# Function for preprocessing images
def preProcessing(img):
    # Check if the image is grayscale
    if len(img.shape) > 2 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # Normalize the image
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# Apply preprocessing to train, test, and validation sets
x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))
x_validation = np.array(list(map(preProcessing, x_validation)))

# Reshape the datasets to add the channel dimension
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

# Create data generator with augmentation
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10)

data_generator.fit(x_train)

# Convert labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=no_of_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=no_of_classes)
y_validation = tf.keras.utils.to_categorical(y_validation, num_classes=no_of_classes)

# Define training parameters
batch_size = 50
epochs = 100

# Calculate steps per epoch dynamically
steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_validation) // batch_size

# Define the model function using TensorFlow/Keras
def my_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(60, (5, 5), input_shape=(image_dim[0], image_dim[1], 1), activation='relu'),
        tf.keras.layers.Conv2D(60, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(30, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(30, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(no_of_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate the model
model = my_model()

# Print model summary
print(model.summary())

# Ensure the dataset repeats
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator.flow(x_train, y_train, batch_size=batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, image_dim[0], image_dim[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, no_of_classes), dtype=tf.float32))
).repeat()

# Create validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(batch_size).repeat()

# Train the model
history = model.fit(train_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=validation_dataset,
                    validation_steps=validation_steps,
                    shuffle=True)

# Plot training and validation loss
plt.figure(1)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.title('Loss')
plt.xlabel('No of Epochs')
plt.show()

# Plot training and validation accuracy
plt.figure(2)
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.title('Accuracy')
plt.xlabel('No of Epochs')
plt.show()

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Save the model
pickle_out = open("model_trained.p", "wb")
try:
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
except Exception as e:
    print(f"Error while saving the model: {e}")

pickle_out.close()
