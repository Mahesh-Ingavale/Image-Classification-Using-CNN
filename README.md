# Image-Classification-Using-CNN  
Transfer Learning - Image Classification [ Artificial Intelligence ]
 
#Part 1
#importing requirements
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split = ['train[:80%]', 'train[80%:90%]', 'train[90:]'],
        with_info=True,
        as_supervised=True,

)

#Part 2
print(raw_train)
print(raw_validation)
print(raw_test)

#Part 3
get_label_name = metadata.features['label'].int2str
for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
  
  #Part 4
  IMG_SIZE= 160
def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5)-1
  image = tf.image.resize(image,(IMG_SIZE, IMG_SIZE))
  return image, label
  
  #Part 5
  # Apply this function to each item in the dataset using the map method
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
BATCH_SIZE=32

#Part 6
#Shuffle and Batch the data
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
SHUFFLE_BUFFER_SIZE = 1000
for image_batch, label_batch in train_batches.take(1):
  pass
image_batch.shape


#Part 7
#MoblieNet V2
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
#Create the base model fro the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
                                               
#Part 8
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#Part 9
#Training the classifier
base_model.trainable = False
#Base model architectre
base_model.summary()

#Part 10
#Generete Predictions
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

#Part 11
#prediction Layer
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#Part 12
model = tf.keras.Sequential([
                             base_model,
                             global_average_layer,
                             prediction_layer
])
base_learning_rate = 0.0001
model.compile(optimizer = tf.optimizers.RMSprop(lr=base_learning_rate),
              loss = tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
model.summary()

#Part 13
len(model.trainable_variables)

#Part 14
#Testing Model
initial_epochs = 10
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))

#Part 15
history = model.fit(train_batches,
                    epochs = initial_epochs,
                    validation_data = validation_batches)
                    
#Part 16
base_model.trainable = True
#Check How many layes in the base model
print(len(base_model.layers))

#Part 17
#fine-tune fro this layer onwords
fine_tune_at = 100
#freeze all the layers before the fine_tune_at layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
model.compile(loss = tf.losses.BinaryCrossentropy(from_logits = True),
              optimizer = tf.optimizers.RMSprop(lr = base_learning_rate/10),
              metrics = [tf.metrics.BinaryAccuracy(threshold = 0.0 , name='accuracy')])

model.summary()
print("initial accuracy: {:.2f}".format(accuracy0))

#Part 18
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2,1,2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc = 'upper right')
plt.ylabel('cross entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#Part 19
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(train_batches,
                         epochs = total_epochs,
                         initial_epoch = history.epoch[-1],
                         validation_data = validation_batches)

#Part 20
# Accuracy And Loss Curves
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8,1])
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(2,1,2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0,1.0])
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epochs')
plt.show()

#End of Code
