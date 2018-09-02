## Traffic Sign Recognition 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./imp_images/histogram.png "Histogram"
[normalised]: ./imp_images/normalised.png "Standarizing"
[newImage1]: ./new_online_images/2-1.jpg "Traffic Sign 1"
[newImage2]: ./new_online_images/1-2.png "Traffic Sign 2"
[newImage3]: ./new_online_images/25-1.png "Traffic Sign 3"
[newImage4]: ./new_online_images/40-1.jpg "Traffic Sign 4"
[newImage5]: ./new_online_images/14-1.jpg "Traffic Sign 5"
[softmax]: ./imp_images/softmax.png "Softmax probabilities"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
All the rubric points are covered in the below writeup.


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You can find my code in the file "Traffic_Sign_Classifier.ipynb". It is properly annoteted.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
In the notebook, in the first 3 blocks of code a summary of the data set shape and number of samples is provided. In addition to plotting random samples of images from the training samples, and a histogram representing the sample count for each individual class.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Training set size is 34799
* Test set size is 12630
* The shape of traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.
 

Visualization of the data set is shown here. It is a bar chart showing how the image data is spread over different image classes.

![alt text][histogram]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I have decided to go with the original RGB images as i think converting image to grayscale caused significant loss of the features, which plays inportant role in the identifying the classed which are dependent on color.

To achieve consistency in the training set of images, I have normalized/standardized the images.

Here is an example of a traffic sign image before and after standardization .

![alt text][normalised]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data is splitted to training and validation sets. (9th cell in IPython Notebook).

I randomly split the training data into a training set and validation set for cross validation. I used train_test_split function from sklearn.model_selection.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 8th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (0) Input         		| 32x32x3 RGB image   							| 
 | (1) Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| (2) RELU					|												|
| (3)Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| (4) dropout					|										0.5		|
| (5) Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| (6) RELU					|												|
| (7) Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| (8) Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400     									|
| (9) RELU					|												|
| (10) Flatten 7 and 9					|									400, 400			|
| (11) Concatenate					|							800					|
| (12) dropout					|										0.5		|
| (13) Fully connected		| outputs 43 classes        									|



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10th cell of the ipython notebook. 

I used an Adam optimizer to train the model,Used batch size of 128, 50 epochs, and 0.001 learning rate.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Accuracy calculation is done in the 12th cell in IPython Notebook.

My final model results are:
* training set accuracy of 99.6%
* validation set accuracy of 99.0% 
* test set accuracy of 93.7%

I have used LeNet model that is there in the lectures after modifying it for RGB images, as i have decided to use all three chanels to get better accuracy. But found that it has some limitations like Overfitting which is causing training data to cause poor validation accuracy. So i have searched for different architectures that are capable of the same tasks. Found one which is not complex but give better aacuracy than LeNet.
I came across the research paper (included in the package "arch.pdf"), studied it and then implemented. there was still an issue of overfitting. In further investigation it came to picture that dropout layer can reduce overfitting. I MOdified the Architecture a bit :
* Changing the filter dimensions to accept RGB input instead of greyscale images.
* Adding a dropout after the first convolution layer to decrease the effect of over-fitting.

After this tuning the parameter can increase the accuracy. Learning rate, batch size, epoch count and keep probability were the parameters that can be tuned.
After tunning, the below parameters were found to yield the best results:
* Learning rate of 0.001
* Batch size of 128
* Epoch count of 30
* Keep probability of 0.5
The addition of dropout layer achieved its aim of reducing the effect of over-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

5 samples from the 25 German traffic signs that I found on the web:

![alt text][newImage1] ![alt text][newImage2] ![alt text][newImage3] 
![alt text][newImage4] ![alt text][newImage5]

All the images are correctly classified from all these 5 images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Results of the prediction are:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h	      		| 50 km/h			 				|
| 30 km/h	      		| 30 km/h				 				|
| Road work			| Road work      							|
| Roundabout     			| Roundabout 										|
| Stop Sign      		| Stop sign   									| 


The model was able to predict 24 of the 25 traffic signs correctly, which evaluates to an accuracy of 96%. This is good if  we compare it to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
The graphs below show the softmax probabilities for the top 5 classes predicted by the model for each image. The model was able to predict 24 of the 25 traffic signs correctly, which evaluates to an accuracy of 96%. Accordingly, this misjudgement of 1 example can be due to the fact that, unlike the training data set, the image is not that clear that too in normalised form to predict it correctly. THe image has been predicted as narrow down to left but is not.this can be the indication of lossed feature data.

![alt text][softmax]