from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.optimizers import Adadelta
from keras.utils import plot_model
from keras import initializers

import numpy as np


#Parameters for the training updates
img_width, img_height = 400, 250
nb_train_samples = 3000 #1184 unique pics
nb_validation_samples = 400 #90 unique pics
batch_size = 32
epochs = 400

label_mode="binary"

train_data_dir="../PICS/trainnew"
validation_data_dir="../PICS/valnew"

base_model = InceptionV3(weights='imagenet',include_top=False,input_shape = (img_height, img_width, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
# We add a fully-connected layer
#x = Dense(1024, activation='relu',kernel_initializer=initializers.VarianceScaling(scale=2.0))(x)

# and a logistic layer
predictions = Dense(1, activation='sigmoid',kernel_initializer=initializers.VarianceScaling(scale=2.0))(x)

model= Model(inputs=base_model.input, outputs=predictions)

#We freeze the model expected the added layers
for layer in base_model.layers:
    layer.trainable = False

#Compilation of the model :

#We use ada-delta
opt=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=10)

test_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=10)

print("Training Set : ")
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = label_mode)

print("Validation Set : ")
validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = label_mode)


for X_batch, y_batch in train_datagen.flow_from_directory(train_data_dir,target_size = (img_height, img_width),batch_size = 2, class_mode = label_mode):
    print(X_batch.shape)
    #print(X_batch)
    plt.imshow(X_batch[0,:,:,:])
    # show the plot
    plt.show()
    print(y_batch)
    break

# Model Callbacks

#Saved every 2 epochs in this case
save_period=2
checkpoint = ModelCheckpoint("Inceptionv3_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=save_period)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Training : (Use Keras 2.0)
model.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples/batch_size,
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples/batch_size,
callbacks = [checkpoint, early])
