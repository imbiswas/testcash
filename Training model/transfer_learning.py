import os
import sys
import glob

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 10
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

def get_nb_files(directory):
  
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def setup_to_transfer_learn(model, base_model):
  
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

def setup_to_finetune(model):

  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

nb_train_samples = get_nb_files('/home/biswas/Documents/Cash Project Model/Cash Data Folder/final_data_extra/data/train_sample')
nb_classes = len(glob.glob("/home/biswas/Documents/Cash Project Model/Cash Data Folder/final_data_extra/data/train_sample/*"))
nb_val_samples = get_nb_files('/home/biswas/Documents/Cash Project Model/Cash Data Folder/final_data_extra/data/valid_sample')
nb_epoch = int(NB_EPOCHS)
batch_size = int(BAT_SIZE)

train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
'/home/biswas/Documents/Cash Project Model/Cash Data Folder/final_data_extra/data/train_sample',
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,
)

test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

validation_generator = test_datagen.flow_from_directory(
'/home/biswas/Documents/Cash Project Model/Cash Data Folder/final_data_extra/data/train_sample',
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,
)

base_model = InceptionV3(weights='imagenet', include_top=False)

model = add_new_last_layer(base_model, nb_classes)

setup_to_transfer_learn(model, base_model)

history_tl = model.fit_generator(
train_generator,
epochs=nb_epoch,
steps_per_epoch=nb_train_samples,
validation_data=validation_generator,
validation_steps=nb_val_samples)

setup_to_finetune(model)

history_ft = model.fit_generator(
train_generator,
steps_per_epoch=nb_train_samples,
epochs=nb_epoch,
validation_data=validation_generator,
validation_steps=nb_val_samples)

model.save('/home/biswas/Documents/Cash Project Model/Model files/sample.model')