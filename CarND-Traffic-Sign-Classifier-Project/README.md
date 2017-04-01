#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./Explore1.png "Visualization"
[image2]: ./hist1.png "Label distriburion"
[image3]: ./hist2.png "Label distriburion"

[preprocess0]: ./preprocess0.png "Preprocessing example"
[preprocess1]: ./preprocess1.png "Preprocessing example"
[preprocess2]: ./preprocess2.png "Preprocessing example"
[preprocess3]: ./preprocess3.png "Preprocessing example"
[preprocess4]: ./preprocess4.png "Preprocessing example"

[heatmap]: ./heatmap.png "Heatmap"
[avgscore]: ./avg_score.png "Average scores"
[accurancy_on_valid] ./accurancy_on_valid.png "Average score on validation dataset while learning"

[Example_of_erros]: ./Example_of_erros.png "Example of errors"

[new_images]: ./new_images.png "Images from the web"
[model_preformance]: ./model_preformance.png "Model performance onImages from the web"

[convolution1_0]: ./convolution1_0.png "First convolution"
[convolution1_1]: ./convolution1_1.png "First convolution"
[convolution1_2]: ./convolution1_2.png "First convolution"
[convolution1_3]: ./convolution1_3.png "First convolution"
[convolution1_4]: ./convolution1_4.png "First convolution"


[convolution2_0]: ./convolution2_0.png "Second convolution"
[convolution2_1]: ./convolution2_1.png "Second convolution"
[convolution2_2]: ./convolution2_2.png "Second convolution"
[convolution2_3]: ./convolution2_3.png "Second convolution"
[convolution2_4]: ./convolution2_4.png "Second convolution"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. Fitst of all let's look on images:

![Images and number of classes][image1]

Now we have provided train/valid/test datasets - we will explore distribution of targer variable across them.

![Distribution of image classes][image2]
![Distribution of image classes2][image3]

We have that distribution of target values are near similar, so we will assume that images are from the same dataset and we do not need to change number of examples per label. UPD: Anyway, I've tried to add a dataset with equal number of images per class.

###Design and Test a Model Architecture

####1. I've tryied three different tecnics of preporcessing:
1) Grayscaling + normalization (see function preprocess, only using numpy for test)
2) Got from a paper: normalization (this time using OpenCv), converting to YUV and using normalization for Y channel (todo: add by-part normalization like in paper)
3) Using skimage Hist

There are few examples of second preprocessing algoritm:

![second variant of preprocessing][preprocess0]
![second variant of preprocessing][preprocess1]
![second variant of preprocessing][preprocess2]
![second variant of preprocessing][preprocess3]
![second variant of preprocessing][preprocess4]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I did not make anything to make new validation set, just used one from pickle. It seems (according to target distribution and final scores) that this one is ok - all train/validation/test are random parts of the same dataset. Ofcourse if we have more time and have gpus we can use cross validation: split our dataset into 3-5 parts and make learning/validating 3-5 times.

I've tryied to create "Jittered dataset" like in the paper - adding random scaling and rotations to original dataset. To each image in train set I've added 4 images with random rotation +-10 and random scaling to 1.1 for x/y axis + random cut-off on the top or bottom size. There are an example of all kind of transformation (for each image I've choosed random 4):

![Image Transofrmations][Jittered]

So, I've increased number of examples in train dataset 5 times using Jittered dataset.

Also, I've tried keras function ImageDataGenerator and using it created 4 different jitered datasets:

1) Using my function, for each image x4 transformed images
2) Using my function, for each class 5000 images 
3) Using keras function, for each image x4 transformed images
4) Using keras function, for each class 5000 images 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I've not really played a lot with parametrs: I've took optimizer, batch size and learning rate from lenet for mninst example and added a dropout layer after second convolution because without it model started overfitting too fast. I've made 16 iterations of learning:

1) For each of 4 datasets
2) For 2 prepocessing methods
3) With dropout with prob=0.5 and without dropout

Result on validation datasets are:

|Dataset|preprocess_v2,  with dropout |preprocess_v2, no dropout|preprocess_v3,  with dropout|preprocess_v3, no dropout|
| -----:|-----:|-----:|-----:|-----:|
|v1|0.971202|0.953061|0.973243|0.964626|
|v2|0.971655|0.948980|0.973243|0.964399|
|v3|0.972109|0.957823|0.971655|0.965533|
|v4|0.969615|0.958730|0.969388|0.964853|

So the best iteration was for preprocess_v3, with dropout and for first and second datasets. This is a heatmap of the results:
![heatmap on validation set][heatmap]

Also, I've looked for performance of individual parametrs in average:

![Average on valid][avgscore]

We can sea, that different datasets do not give as big performance changes, but preprocessing_v3 was really better then v_2 in average and using dropout really helps. I've also tried to compare using v3 and v1 datasets (see in notebook), but it looks like v_1 really better. Other thing I've learned from learing curve is that model is underfitting and I need more iterations. For best parametrs I've trained it up to 300 iterations and got this learning curve:

![Learning curve on train/validateion set][accurancy_on_valid]

Since validation dataset is tiny and sometimes we see outliers I've added moving average and selected 85 iteration as the best in moving average. Than I trained model for 85 epochs and checked the performance on test dataset - in was 0.95, so enough for now ;) But we can see some overfitting: on validation dataset it was 0.968 on moving average.

### Performance on the test dataset:

In first, we see that for some categories accurancy is very low and there are a correlation between accurancy for label and %of images in training dataset (>0.3), so for better model we need more data for rear categories. 

Also, I've provided information confusion matrix and more detailed examples of classes where model makes mistake:

![Example of errors][Example_of_erros]

We can sea that for many images with mistakes we have too low quality, so for better model we need better image quality and more examples. 
 
###Test a Model on New Images

####1. Here are five German traffic signs that I found on the web. I've cropped only traffic sign and resized them to 32x32:

![Images from the web][new_images] 

I've calculated probabilities for each class, and I've one mistake so the accurancy is 80%

![Model performance on new images][model_preformance]

Actually I don't know what is the problem with third image - my guess is that it is not from Germany ;)

### Analysis of convolution layers

I've provided images of model performance on convolution layers

![First convlolution][convolution1_0]
![First convlolution][convolution1_1]
![First convlolution][convolution1_2]
![First convlolution][convolution1_3]
![First convlolution][convolution1_4]


![Second convolution][convolution2_0]
![Second convolution][convolution2_1]
![Second convolution][convolution2_2]
![Second convolution][convolution2_3]
![Second convolution][convolution2_4]


On the first layer we can see our signs, on the second I can't see anything ;) I've tried to look on this laerys for images with mistake on test dataset, but I do not have any idea how to use this information to change model structure.
