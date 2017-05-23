

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[bboxes]: ./output_images/bbox.png
[examples]: ./output_images/examples.png
[hog]: ./output_images/hog.png
[t1]: ./output_images/test1.png
[t2]: ./output_images/test2.png
[t3]: ./output_images/test3.png
[t4]: ./output_images/test4.png
[t5]: ./output_images/test5.png
[t6]: ./output_images/test6.png

[similar0]: ./output_images/similar_0.png
[similar1]: ./output_images/similar_1.png
[similar2]: ./output_images/similar_10.png

[dist]: ./output_images/dist.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README



### Histogram of Oriented Gradients (HOG)

#### 0. First of all let's look on images from dataset:

![Example images][examples]

We can see, that we have a lot of similar images of the car in our dataset. Of course, when we will make cross validation and will try to choose estimator hypermarametrs it will tend to overfitting. To prevent it the best thing (since similar images are in the row) is to calculate (for example) histogram features and calculate distances between images in a row. Wi will have the following image:

![distance][dist]

So, when the image changes (new car) we will have a peak in our plot. If we split the plot by this peaks wi will have different cars, like this:

![Example images][similar0]
![Example images][similar1]
![Example images][similar2]

But I've not used this split to different classes - just SVM on all images worked OK and I do not have time to make something better:)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
#### 2. Explain how you settled on your final choice of HOG parameters.
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

(see first part of notebook, it's commented well!)

At first cells of notebook I used functions from lessong to make features. I've experimented a bit, but the best parametres as I found (just with stratified kFold, without spliting by classes like as 0 were

color_space =   YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
classifier - linear SVM witch C=1 (I've not played with C - but score 0.991 is enough:))

This is an example of hog features in YCrCb for car and not car

![hog_features][hog]

Even better score (significantly) is for ensemble of models for different colorspaces - but I've no need to use it. Good thing that it could be calculated in parallel. 

After choosing this parametres I've fitted classifier on all data.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Goal is to make function, which will make hog transformation of whole image and then take hog features just of the needed part. It will be fast (not really - 1 image per second as a result). 

My function  take an part of image (from ystart to ystop) than scale it (we've scale variable) and than calculate needed number of "boxes" with fixed step measured in hog's cell (it's needed because we are going to calculate hog only once)

I've used three scales - 1, 1.5 and 2 and a little bit different parts of image to speed up:

![bboxes][bboxes]


For all features I've calculated predictions, than made a heatmap and applyied threshold <=1. Then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. 

I've found my pipeline works good on test images, see the examples:

![bboxes][t1]
![bboxes][t2]
![bboxes][t3]
![bboxes][t4]
![bboxes][t5]
![bboxes][t6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For avoiding false positives I've used median of heatmaps last 5 frames of video, before using `scipy.ndimage.measurements.label()` - so I need at least 3 mistakes in 5 frames to make a false box. The bad thing is that for fast driving cars it could be a delay in borders of image. Fore details see notebook.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* First of all I've lack of time now - so advanced features are not done.
* For making better predictions we have to use better CV approach and more images from different datasets.
* At first we can make feature selection better. Work witjh individual channels for hog, histograms etc - it will giv us possility to make feature extraction much faster (if use less features in classifier)
* We have to try different classifiers (if not using NN at least use XGB :)) and try to play with hyperparametrs
* It's possible to make features and predictions in parallel - it will speedup calculation
* Finally I really interesed to study NN approach to this task ang going to try some models like YOLO to do it.

# Thanks for a great task!)
