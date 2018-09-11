# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center"
[image2]: ./images/left.jpg "Left"
[image3]: ./images/right.jpg "Right"
[image4]: ./images/hsv.png "HSV"
[image5]: ./images/noise.png "Noise"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_track_1.h5 containing a trained convolution neural network for track 1
* model_track_2.h5 containing a trained convolution neural network for track 2
* writeup_report.md or writeup_report.pdf summarizing the results

The project has been developed on Windows 10 with tensorflow v1.3.0, keras v2.2.2 and cuda v9.0.176.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing.

```sh
python drive.py model_track_1.h5
```

or for track 2

```sh
python drive.py model_track_2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a convolution neural network that consists of 4 convolution layer of size 10, 20, 40 and 80 (model.py line 33-40) and two dense layers of size 1024 (model.py line 45-48). MaxPooling was used to reduce the the images size and at the end of the convolution layer global MaxPooling was used (model.py line 43). To prevent overfitting dropout was added before the dense layers.
All convolution layers are using RELU as activation function.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 45, 47). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. The best result was achieved by using the default learning rate at 0.0001 (model.py line 180).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For the first track all images was augmented by flip in the generator (model.py line 125). For the second track flip did not make sense because the car should drive in the right lane. Therefore all track 2 images was augmented by adding gaussian  noise to imitate shaded areas.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At the beginning I followed the strategy from the project lecture. The result had to be improved and I tried to use transfer learning by using Keras build in networks. I failed hard because the prediction time was to slow to correct the car in real time.

Next I tried to build a deeper convolutional network from scratch with a cropping and normalization layer, it worked much better. I also used images from left and right camera and adjusted the correction coefficient. Then I added a Lambda layer to convert RGB image to HSV and stacked it with a gray image, the result was improved. 

![alt text][image4]
RGB image and the HSV channels

But still I had some problems with spots where the vehicle fell off the track. To fix it I recorded some extra data in the difficult regions and overfitted the model a bit, but it worked.

The model had lot of problems with the second track, so I added an extra convolution layer and add more nodes to the dense layer. At the same time I also augmented the images by adding gaussian noise. The mean squared error were much higher than for track 1 and might be caused by the capacity of the model is too small or because by bad driving and collection of inconsistent data.

At the end of the process, the vehicle was able to drive autonomously around the track 1 without leaving the road. Improvements is still needed for track 2. 


#### 2. Final Model Architecture

The final model architecture (model.py lines 27-51) consisted of a convolution neural network with the following layers.

* a Keras lambda layer is used to crop the image (model.py line 29)
* a Keras lambda layer is used to convert the image to gray and hsv colorspace and combined (model.py line 30)
* 3 groups of convolution layer + a MaxPooling layer, with 5x5 filters of size 10, 20, 40 (model.py line 33-40)
* 1 convolution layer + a global MaxPooling layer, with 3x3 filter of size 80 (model.py line 42-43)
* 2 groups of a dropout + a dense layers of size 1024 (model.py line 45-48)
* 1 dense layer of size 1 (model.py line49)

All convolution layers are using RELU as activation function, except the dense layers.

PS: I realized that dense layer does not have any activations, so no non-linearity in the dense layers and all could be substituted with the last dense layer. But the layers have dropout and I don't know how to replace it correct. If I add the RELU as activation for the 2 first dense layers track 1 will fail. 

|Layer (type)                 | Output Shape              | Param # |  
|-------------------------------------------------------------------|
|cropping2d_7 (Cropping2D)    | (None, 90, 320, 3)        | 0       |
|lambda_13 (Lambda)           | (None, 90, 320, 4)        | 0       |       
|lambda_14 (Lambda)           | (None, 90, 320, 4)        | 0       |         
|conv2d_25 (Conv2D)           | (None, 86, 316, 10)       | 1010    |      
|max_pooling2d_19 (MaxPooling)| (None, 43, 158, 10)       | 0       |         
|conv2d_26 (Conv2D)           | (None, 39, 154, 20)       | 5020    |      
|max_pooling2d_20 (MaxPooling)| (None, 19, 77, 20)        | 0       |         
|conv2d_27 (Conv2D)           | (None, 15, 73, 40)        | 20040   |     
|max_pooling2d_21 (MaxPooling)| (None, 7, 36, 40)         | 0       |         
|conv2d_28 (Conv2D)           | (None, 5, 34, 80)         | 28880   |     
|global_max_pooling2d_7       | (None, 80)                | 0       |         
|dropout_15 (Dropout)         | (None, 80)                | 0       |         
|dense_19 (Dense)             | (None, 1024)              | 82944   |     
|dropout_16 (Dropout)         | (None, 1024)              | 0       |         
|dense_20 (Dense)             | (None, 1024)              | 1049600 |   
|dense_21 (Dense)             | (None, 1)                 | 1025    | 

Total params: 1,188,519
Trainable params: 1,188,519
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

Then I used the left and right images with a correction koeficient on 0.17, it helped to steer the car to center of the lane. 

![alt text][image2]
![alt text][image3]

I flipped the images and recorded extra data bur driving the opposite direction. In the difficult areas I recorded some extra data.

For track 2 I recorded 1 lap and augmented images by adding gaussian noise to 

![alt text][image5]

All augmentation was implemented into the generator.

By monitoring the loss for track 1 I selected epocs 20 to prevent overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. For track 2 I has to increase the epocs to 100.




