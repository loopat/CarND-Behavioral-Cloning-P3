
**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


**Rubric Points**

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

**Files Submitted & Code Quality**

***1. Submission includes all required files and can be used to run the simulator in autonomous mode***

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.ipynb summarizing the results

***2. Submission includes functional code***

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

***3. Submission code is usable and readable***

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Model Architecture and Training Strategy**

*<1. Collect Trainning Data*

To capture good driving behavior, I first recorded the images on track one using center lane driving. 

Here is an example image of center lane driving:

![center.jpg](attachment:center.jpg)

*<2. Enlarge Trainning Data*

To englarge training data, I also used left images and right images generated during the training mode with adjustment and also filpped them.

The following are the right images and left images:

![right.jpg](attachment:right.jpg)

![left.jpg](attachment:left.jpg)

Ref to line 24 - 53 in model.py

*<3. Normalization and Crop the images*

A lambda layer is used to parallelize image normalization in line 72 in model.py.

And then crop the images to focus only the part of images uesful for predicting a steering angle.

*<4. NetWork

Here, I adopted the architecture published by autonomous vehicle team at NVIDIA. 

I thought this model might be appropriate because both of them are in the similar case to make the car drive autonomously.

- My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64, refer to model.py lines 78-82) 

- The model includes RELU layers to introduce nonlinearity (code lines 78-82), and the data is normalized in the model using a Keras lambda layer (code line 72). 

- The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).


```python

```
