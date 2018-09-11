import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization, Dropout, AlphaDropout, Lambda
from keras.models import Sequential
from keras.optimizers import Adam

# path to data and model
data_path = './data/car-sim1/'
img_path = data_path + 'IMG/'
driving_log_path = data_path + 'driving_log.csv'
model_path = 'model.h5'

# model helper function to convert and stack images (use hsv colorspace and gray)
def stack(x):
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    return tf.stack([hsv[:,:,:,0], hsv[:,:,:,1], hsv[:,:,:,2], gray[:,:,:,0]], axis=-1) 

# create the model
def createModel():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(stack))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(10, (5,5), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(20, (5,5), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(40, (5,5), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(80, (3,3), activation='relu'))
    model.add(GlobalMaxPooling2D())

    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(1))

    return model

# read log file
def read_log(file):
    samples = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

# augment image by add gaussian noise
def add_gaussian_noise(images, blend=0.25):
    gaussian_noise_imgs = []
    row, col, _ = images[0].shape
    
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for image in images:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)*255
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(image.astype(np.float32), 1-blend, blend * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
        
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.uint8)
    return gaussian_noise_imgs

# load RGB image
def loadImageRgb(imagePath):
    image = cv2.imread(imagePath)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# the image generator
def generator(samples, batch_size=32, augment=False, flip=True, noice=False):
    
    images = []
    angles = []
    
    count = 0

    while 1: # Loop forever so the generator never terminates
        samples = np.random.permutation(samples)
        
        for sample in samples:
            
            augment_images = []
            augment_angles = []
                
            correction = 0.17
            angle = float(sample[3])

            # center image
            filename = img_path+sample[0].split('/')[-1]
            image = loadImageRgb(filename)
            augment_images.append(image)
            augment_angles.append(angle)

            if augment:
                
                # left image
                filename = img_path+sample[1].split('/')[-1]
                image = loadImageRgb(filename)
                augment_images.append(image)
                augment_angles.append(angle + correction)

                # right image
                filename = img_path+sample[2].split('/')[-1]
                image = loadImageRgb(filename)
                augment_images.append(image)
                augment_angles.append(angle - correction)
                
                if flip:
                    # flip images
                    flip_images = []
                    flip_angles = []
                    for image, angle in zip(augment_images, augment_angles):
                        flip_images.append(cv2.flip(image,1))
                        flip_angles.append(angle*-1.0) 
                    augment_images.extend(flip_images)
                    augment_angles.extend(flip_angles)
                
                if noice:
                    # add noise
                    noise_images = add_gaussian_noise(augment_images, blend=0.25)
                    noise_angles = augment_angles
                    augment_images.extend(noise_images)
                    augment_angles.extend(noise_angles)
      
            images.extend(augment_images)
            angles.extend(augment_angles)
                     
            while len(images) >= batch_size:

                X_train = np.array(images[:batch_size])
                y_train = np.array(angles[:batch_size])
                images = images[batch_size:]
                angles = angles[batch_size:]

                yield sklearn.utils.shuffle(X_train, y_train)



# meta parameters
epochs = 20
batch_size = 32
learning_rate = 0.0001
test_size = 0.2

# load train and validation set (no test set is loaded, the real proof is the simulator)
print('read samples...')
samples = read_log(driving_log_path)
train_samples, validation_samples = train_test_split(samples, test_size=test_size)

# initial generators
train_generator = generator(train_samples, batch_size=batch_size, augment=True)

# do not augment validation set, validation set should have same distribution as real world
validation_generator = generator(validation_samples, batch_size=batch_size) 

# calculate steps_per_epoch and validation_steps for fit_generator
steps_per_epoch = int(np.ceil((len(train_samples)*3*2)/batch_size)) # 3x for center, left and right, 2x for flip
validation_steps = int(np.ceil(len(validation_samples)/batch_size))

# create and compile the model
print('create and compile model...')
model = createModel()
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# train the model using the generator function
print('train model...')
model.fit_generator(train_generator, 
    steps_per_epoch=steps_per_epoch, 
    validation_data=validation_generator, 
    validation_steps=validation_steps, 
    epochs=epochs
)

# save the model in models
print('save model as: ' + model_path)
model.save(model_path)