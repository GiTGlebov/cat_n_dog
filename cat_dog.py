from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import shutil
import os

data_dir = 'C://Downloads/train'
data_dir2 = 'C://Downloads/PetImages/Dog'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
os.chdir("D://ds")
os.getcwd()
print(data_dir)


def create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "cats"))
    os.makedirs(os.path.join(dir_name, "dogs"))


'''create_dir(train_dir)
create_dir(val_dir)
create_dir(test_dir)'''


def copy_images(start_i, end_i, source_dir, dest_dir):
    for i in range(start_i, end_i):
        shutil.copy2(os.path.join(source_dir, 'cat (' + str(i) + ')' + ".jpg"), os.path.join(dest_dir, "cats"))
        shutil.copy2(os.path.join(source_dir, 'dog (' + str(i) + ')' + ".jpg"), os.path.join(dest_dir, "dogs"))


'''copy_images(1, 9001, data_dir, train_dir)
copy_images(9001, 10501, data_dir, val_dir)
copy_images(10501, 12301, data_dir, test_dir)'''

img_w = 150
img_h = 150
input_shape = (img_w, img_h, 3)
batch = 100
epochs = 10
traink = 18000
testk = 3600
valk = 3000

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

data_gen = ImageDataGenerator(rescale=1. / 255)

train_gen = data_gen.flow_from_directory(train_dir, target_size=(img_w, img_h), batch_size=batch, class_mode='binary')
val_gen = data_gen.flow_from_directory(val_dir, target_size=(img_w, img_h), batch_size=batch, class_mode='binary')
test_gen = data_gen.flow_from_directory(test_dir, target_size=(img_w, img_h), batch_size=batch, class_mode='binary')

history = model.fit(train_gen, steps_per_epoch=traink // batch, epochs=epochs, validation_data=val_gen,
          validation_steps=valk // batch)

scores = model.evaluate_generator(test_gen, testk // batch)
print(scores[1] * 100)

