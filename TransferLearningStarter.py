import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()

IMG_SIZE = 128
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img
TRAIN_PATH = '../dl2_trimages/'
TEST_PATH = '../dl2_teimages/'
img = read_img(TRAIN_PATH+'scan_0000.png')
img.shape

train_img = []
for img_name in tqdm(train['image_name'].values):
    train_img.append(read_img(TRAIN_PATH + img_name))
print("done")
x_train = np.array(train_img, np.float32) / 255.
print("del")
del train_img
x_train.shape
print("shape",x_train.shape)
label_list = train['detected'].tolist()
label_numeric = {k: v+1 for v, k in enumerate(set(label_list))}
print("enum")
y_train = [label_numeric[k] for k in label_list]
print("ytrain");
y_train = np.array(y_train)
print("1");
from keras import applications
print("2");
from keras.models import Model
print("3");
from keras import optimizers
print("4");
from keras.models import Sequential
print("5");
from keras.layers import Dense, Flatten
print("6");
from keras.utils import to_categorical
print("7");
from keras.preprocessing.image import ImageDataGenerator
print("8");
y_train = to_categorical(y_train)
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
batch_size = 32 # tune it
epochs = 25 
# increase it lb:0.40856(50 epochs), 0.27101 (5 epochs)
# increase it only on the cloud so that resource allocation error doesnt occur.

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1)
train_datagen.fit(x_train)
history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs
)
del x_train
test_img = []
for img_name in tqdm(test['image_name'].values):
    test_img.append(read_img(TEST_PATH + img_name))
    
x_test = np.array(test_img, np.float32) / 255.
del test_img
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in label_numeric.items()}
pred_labels = [rev_y[k] for k in predictions]
sub = pd.DataFrame({'row_id':test.row_id, 'detected':pred_labels})
sub = sub[['row_id', 'detected']]
filename = 'sub_vgg.csv'
sub.to_csv(filename, index=False)
sub.head()
from IPython.display import FileLink
FileLink(filename)
