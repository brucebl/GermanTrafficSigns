import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
import PIL.Image
import random
import tensorflow as tf
#from google.colab import drive
from tensorflow import keras
from tensorflow.keras import layers

# CNN using Tensorflow to predict traffic signs based on the German Traffic Sign Dataset

# Set up training and test data
#drive.mount("/content/drive")
#dataDir = "/content/drive/MyDrive/Datasets/GermanTrafficSign"
dataDir = "/home/machlearn/Datasets/GermanTrafficSigns"
trainDir = dataDir + "/train"
testDir = dataDir + "/test"
saveWeightsDir = "training_1"
saveModelDir = "save_models"

# Load weights
loadWeights = False
weightsFile = "training_1/save_epoch_15.h5"

# Number of images to test
numTestImgs = 16

# Count the number of classes
numClasses = len(os.listdir(trainDir))

print("Number of classes found: %d" % numClasses)

# List the classes to be inferred
classes = {0:'Speed limit (20km/h)',
          1:'Speed limit (30km/h)',
          2:'Speed limit (50km/h)',
          3:'Speed limit (60km/h)',
          4:'Speed limit (70km/h)',
          5:'Speed limit (80km/h)',
          6:'End of speed limit (80km/h)',
          7:'Speed limit (100km/h)',
          8:'Speed limit (120km/h)',
          9:'No passing',
          10:'No passing veh over 3.5 tons',
          11:'Right-of-way at intersection',
          12:'Priority road',
          13:'Yield',
          14:'Stop',
          15:'No vehicles',
          16:'Veh > 3.5 tons prohibited',
          17:'No entry',
          18:'General caution',
          19:'Dangerous curve left',
          20:'Dangerous curve right',
          21:'Double curve',
          22:'Bumpy road',
          23:'Slippery road',
          24:'Road narrows on the right',
          25:'Road work',
          26:'Traffic signals',
          27:'Pedestrians',
          28:'Children crossing',
          29:'Bicycles crossing',
          30:'Beware of ice/snow',
          31:'Wild animals crossing',
          32:'End speed + passing limits',
          33:'Turn right ahead',
          34:'Turn left ahead',
          35:'Ahead only',
          36:'Go straight or right',
          37:'Go straight or left',
          38:'Keep right',
          39:'Keep left',
          40:'Roundabout mandatory',
          41:'End of no passing',
          42:'End no passing veh > 3.5 tons'
          }

classKeysStrList = list(map(str, classes.keys()))
print(classKeysStrList)

# Check classes found equals the classes defined
if not numClasses == len(classes):
 print ("Error, the number of training directories not equal to defined classes!")

# Gather training directories and count images
dirs = os.listdir(trainDir)

trainNumAry = []
classNumAry = []

for dir in dirs:
 trainFiles = os.listdir(trainDir + "/" + dir)
 trainNumAry.append(len(trainFiles))
 classNumAry.append(classes[int(dir)])

# Sort by number of images
zipLists = zip(trainNumAry, classNumAry)
sortedPairs = sorted(zipLists, reverse=True)
tuples = zip(*sortedPairs)
trainNum, classNum = [list(tuple) for tuple in tuples]

# Plot number of images
plt.figure(figsize=(21, 10))
plt.bar(classNum, trainNum)
plt.xticks(classNum, rotation='vertical')
plt.show()

# Size the images for training
imgHeight = 64
imgWidth = 64
channels = 3

# Batch size to use for training
batchSize = 16

print(classes)

# Training dataset
trainDs = tf.keras.preprocessing.image_dataset_from_directory(
   trainDir,
   labels = "inferred",
   label_mode = "int",
   class_names = classKeysStrList,
   validation_split = 0.2,
   seed = 1337,
   subset = "training",
   image_size = (imgWidth, imgHeight),
   batch_size = batchSize,
)

# Validation dataset
validDs = tf.keras.preprocessing.image_dataset_from_directory(
   trainDir,
   labels = "inferred",
   label_mode = "int",
   class_names = classKeysStrList,
   validation_split = 0.2,
   seed = 1337,
   subset = "validation",
   image_size = (imgWidth, imgHeight),
   batch_size = batchSize,
)

# Show shape of dataset
for imgBat, labBat in trainDs:
   print("Train dataset shape: %s" % imgBat.shape)
   print("Train label shape: %s" % labBat.shape)
   break

for imgBat, labBat in validDs:
   print("Validation dataset shape: %s" % imgBat.shape)
   print("Validation labels shape: %s" % labBat.shape)
   break

# Add randomness to the images
dataAug = keras.Sequential(
   [layers.RandomRotation(factor=0.07)]
)

# Visualize a few images
plt.figure(figsize=(15,10))
for images, labels in trainDs.take(1):
 print(labels)
 augImgs = dataAug(images)
 for i in range(16):
   ax = plt.subplot(4, 4, i+1)
   plt.imshow(augImgs[i].numpy().astype("uint8"))
   label = labels[i].numpy()
   print(label)
   classLabel = classKeysStrList[label]
   print(classLabel)
   classname = classes[int(classLabel)]
   print(classname)
   plt.title("%s:%s" % (classLabel, classname))
   plt.axis("off")

plt.show()

# Function to construct model
def make_model(input_shape, num_classes):
   """Creates the Keras model.
   
   The general architecture is the first layers are convolutional layers
   to identify features in the input image with a pooling layer between them.
   The ending layers are a fully connected layer and then a probability output
   layer for each of the classes.
   """
   
   inputs = keras.Input(shape=input_shape)

   # Augment images
   layerStack = dataAug(inputs)

   # Scale RGB values down to decimal
   layerStack = layers.Rescaling(1.0 / 255)(layerStack)

   # Convolution layer
   layerStack = layers.Conv2D(64, 5, padding="same")(layerStack)
   layerStack = layers.BatchNormalization()(layerStack)
   layerStack = layers.Activation("relu")(layerStack)
 

   # Max pool layer
   layerStack = layers.MaxPool2D(pool_size=(2,2),padding="same")(layerStack)

   # Random dropout nodes
   #layerStack = layers.Dropout(0.2)(layerStack)

   # Convolution layer
   layerStack = layers.Conv2D(128, 3, padding="same")(layerStack)
   layerStack = layers.BatchNormalization()(layerStack)
   layerStack = layers.Activation("relu")(layerStack)

   # Depthwise Convolution layer
   #layerStack = layers.DepthwiseConv2D(
   #    kernel_size=3,
   #    strides=1,
   #    padding="same",
   #    activation="relu"
   #)(layerStack)

   # Max pool layer
   layerStack = layers.MaxPool2D(pool_size=(2,2),padding="same")(layerStack)

   # Flatten layer
   layerStack = layers.Flatten()(layerStack)

   # Random dropout nodes
   #layerStack = layers.Dropout(0.25)(layerStack)

   # Fully connected layer
   layerStack = layers.Dense(128, 
   kernel_initializer='ones',
   #kernel_regularizer=tf.keras.regularizers.L2(0.01),
   activation="relu")(layerStack)

   # Output layer
   layerStack = layers.BatchNormalization()(layerStack)
   layerStack = layers.Dense(num_classes, activation="softmax")(layerStack)

   return keras.Model(inputs, layerStack)

# Build model
model = make_model(input_shape=(imgWidth, imgHeight) + (3,), num_classes=len(classes))

# Plot model diagram
#keras.utils.plot_model(model, show_shapes=True)

# Model achitecture
model.summary()

# Call back functions
#callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),]
#checkpoint_path = "training_1/cp.ckpt"
checkpoint_path = saveWeightsDir + "/save_epoch_{epoch}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Compile model
model.compile(
   optimizer=keras.optimizers.Adam(learning_rate=1e-2),
   loss="sparse_categorical_crossentropy",
   metrics=["accuracy"],
)

# Load previous weights
if loadWeights is True:
   print("Loading weights file: %s" % weightsFile)
   model.load_weights(weightsFile)

# Epochs
epochs = 15

# Train model
history = model.fit(
   trainDs, epochs=epochs, callbacks=[cp_callback], validation_data=validDs,
)

# Save model not working. Will be fixed by TF.
#model.save(saveModelDir, save_format='tf')

# Plot training results
print(history.params)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show(block=True)

plt.figure(figsize=(10,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.show(block=True)

images = os.listdir(testDir)
numFoundImgs = len(images)
print("Numer of test images: %d" % numFoundImgs)

plt.figure(figsize=(15,10))
plt.title("Predictions on %s images." % (numTestImgs))
for index in range(numTestImgs):
    randIndex = random.randrange(numFoundImgs)
    imgPath = testDir + "/" + images[randIndex]
    print(imgPath)
    testImg = tf.keras.preprocessing.image.load_img(imgPath, target_size=(imgHeight, imgWidth))
    aryImg = tf.keras.preprocessing.image.img_to_array(testImg)
    #aryImg = aryImg/255.0
    imgBatch = np.expand_dims(aryImg, axis=0)
    
    ax = plt.subplot(4, 4, index+1)
    plt.imshow(testImg)
    pred = model.predict(imgBatch)
    print("Predicted %s for %s" % (pred, images[randIndex]))
    classname = np.argmax(pred)
    plt.title("%3.1f%% %s: %s" % (np.max(pred)*100, classname, classes[int(classname)]))
    plt.axis("off")

plt.show()