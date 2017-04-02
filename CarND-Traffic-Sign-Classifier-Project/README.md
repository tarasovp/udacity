#**Traffic Sign Recognition** 

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

[image1]: ./images/Explore1.png "Visualization"
[image2]: ./images/hist1.png "Label distriburion"
[image3]: ./images/hist2.png "Label distriburion"

[preprocess0]: ./images/preprocess0.png "Preprocessing example"
[preprocess1]: ./images/preprocess1.png "Preprocessing example"
[preprocess2]: ./images/preprocess2.png "Preprocessing example"
[preprocess3]: ./images/preprocess3.png "Preprocessing example"
[preprocess4]: ./images/preprocess4.png "Preprocessing example"
[Jittered]: ./images/Jittered.png "Example of adding new images"

[heatmap]: ./images/heatmap.png "Heatmap"
[avgscore]: ./images/avg_score.png "Average scores"
[accurancy_on_valid]: ./images/accurancy_on_valid.png "Average score on validation dataset while learning"

[Example_of_erros]: ./images/Example_of_erros.png "Example of errors"

[new_images]: ./images/new_images.png "Images from the web"
[model_preformance]: ./images/model_preformance.png "Model performance onImages from the web"

[convolution1]: ./images/convolution1.png "First convolution"

[convolution2]: ./images/convolution2.png "Second convolution"

[lenet]: ./images/lenet.png "Neural network"



### Data Set Summary & Exploration

####1. Provide a basic summary of the data set 

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.


Here is an exploratory visualization of the data set. Fitst of all let's look on images:

![Images and number of classes][image1]

Now we have provided train/valid/test datasets - we will explore distribution of targer variable across them.

![Distribution of image classes][image2]
![Distribution of image classes2][image3]

We have that distribution of target values are near similar, so we will assume that images are from the same dataset and we do not need to change number of examples per label. UPD: Anyway, I've tried to add a dataset with equal number of images per class.

### Design and Test a Model Architecture

#### 1. I've tryied three different tecnics of preporcessing:
1) Grayscaling + normalization (see function preprocess, only using numpy for test)
2) Got from a paper: normalization (this time using OpenCv), converting to YUV and using normalization for Y channel (todo: add by-part normalization like in paper)
3) Using skimage Hist

There are few examples of second preprocessing algoritm:

![second variant of preprocessing][preprocess0]
![second variant of preprocessing][preprocess1]
![second variant of preprocessing][preprocess2]
![second variant of preprocessing][preprocess3]
![second variant of preprocessing][preprocess4]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I did not make anything to make new validation set, just used one from pickle. It seems (according to target distribution and final scores) that this one is ok - all train/validation/test are random parts of the same dataset. Ofcourse if we have more time and have gpus we can use cross validation: split our dataset into 3-5 parts and make learning/validating 3-5 times.

I've tryied to create "Jittered dataset" like in the paper - adding random scaling and rotations to original dataset. To each image in train set I've added 4 images with random rotation +-10 and random scaling to 1.1 for x/y axis + random cut-off on the top or bottom size. There are an example of all kind of transformation (for each image I've choosed random 4):

![Image Transofrmations][Jittered]

So, I've increased number of examples in train dataset 5 times using Jittered dataset.

Also, I've tried keras function ImageDataGenerator and using it created 4 different jitered datasets:

1) Using my function, for each image x4 transformed images
2) Using my function, for each class 5000 images 
3) Using keras function, for each image x4 transformed images
4) Using keras function, for each class 5000 images 


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten | Output 400 |
| Fully connected | output 120|
| RELU					|												|
| Dropout, prob=0.5		|												|
| Fully connected | output 80|
| RELU					|												|
| Fully connected | output 43|
| RELU					|												|

![Network][lenet]


#### 4. Describe how, and identify where in your code, you trained your model. 
#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. 

I've not really played a lot with parametrs: I've took optimizer, batch size and learning rate from lenet for mninst example and added a dropout layer after second convolution because without it model started overfitting too fast. I've made 16 iterations of learning:

1) For each of 4 datasets
2) For 2 prepocessing methods
3) With dropout with prob=0.5 and without dropout

Result on validation datasets are:

|Dataset|preprocess_v2,  with dropout |preprocess_v2, no dropout|preprocess_v3,  with dropout|preprocess_v3, no dropout|
| -----:|-----:|-----:|-----:|-----:|
|v1|0.983673|0.963719|0.967800|0.961224|
|v2|0.976190|0.970522|0.974150|0.966667|
|v3|0.935828|0.904082|0.937868|0.907029|
|v4|0.900000|0.854875|0.866440|0.847619|


So the best iteration was for preprocess_v2 (my function), with dropout and for first and second datasets. This is a heatmap of the results:
![heatmap on validation set][heatmap]

Also, I've looked for performance of individual parametrs in average:

![Average on valid][avgscore]

We can sea, that see:
1) In average second dataset is the best, but on first we got better performance
2) preprocessing_v2 better than preprocessing_v3
3) dropout really helps

For best parametrs I've trained it up to 200 iterations and got this learning curve:

![Learning curve on train/validateion set][accurancy_on_valid]

Since validation dataset is tiny and sometimes we see outliers I've added moving average. We can see that model is underfitted, but it's not a good idea to make any more iterations - in previous set on 50 iterations we got better accurancy (0.983) - for better tuning we have to use kfold cross-validation or just bigger dataset

### Performance on the test dataset: 0.953

In first, we see that for some categories accurancy is very low and there are a correlation between accurancy for label and %of images in training dataset (>0.3), so for better model we need more data for rear categories. 

Also, I've provided information confusion matrix and more detailed examples of classes where model makes mistake:

![Example of errors][Example_of_erros]

We can sea that for many images with mistakes we have too low quality, so for better model we need better image quality and more examples. 
 
### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web. I've cropped only traffic sign and resized them to 32x32:

![Images from the web][new_images] 

I've calculated probabilities for each class, and I've one mistake so the accurancy is 80%

![Model performance on new images][model_preformance]

Actually I don't know what is the problem with third image - my guess is that it is not from Germany ;)

### Analysis of convolution layers

I've provided images of model performance on convolution layers

![First convlolution][convolution1]

![Second convolution][convolution2]


On the first layer we can see our signs, on the second I can't see anything ;) I've tried to look on this laerys for images with mistake on test dataset, but I do not have any idea how to use this information to change model structure.
