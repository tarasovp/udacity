## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/distortion.png "Undistorted"
[image2]: ./images/undistorted_track.pngg "Road Transformed"
[image3]: ./images/bin_5.png "Binary Example"
[image4]: ./images/projected.png "Warp Example"
[image5]: ./images/visualized_6.png "Fit Visual"
[image6]: ./images/sample.png "Output"
[video1]: ./project_anotated.mp4 "Project video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "explore.ipynb" 

I've nothing to add:

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Example of unfistroeted images][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I've used perspective transform before binaryzation. Unfortanutely I've not seen that sample src and dst are proveided, so I've used my own hardcoded points


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 525, 500      | 250, 250        | 
| 765, 500      | 1030, 250      |
| 1045, 680     | 1030, 690      |
| 260, 680      | 250, 690        |



The code for my perspective transform includes a function called `pr()`, which appears in the 8-10rd code cells of the IPython notebook).  The `pr()` function takes as inputs an image (`img`), and uses global (`src`) and destination (`dst`) variables.  

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've a littlebit modified functions from udacity course, made them to accept few more parametrs, like, for example on which colorspace ('h' - is H from hls, 'A' is A from LAB and 'gray' is just gray:)) to apply them. This gived me a change to do some experiments and to introduce three functions with different threshholds. In first function I use additionally white and yellow points detection from first project.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I've tried both convolution and sliding window approaches but stopped on sliding window. Of coarse, it's better to implement both and have a choice - if first one not working use the second. Also, I've hardcoded the place where lane's should start - it work fine for project and challenge video, but it will not work for harder challenge at all.

![alt text][image5]

When the lane is detected I just look for points adjasting to the previously detected lines and use them to fit polynom. If I've missed 5+ imagess in a row I'm starting from scratch. For mor detailed information please see line class in explore.ipynb.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

It's done in line class. For curvation I just use the same idea as in Udacity course lectures, for the mid point I calculate left and right position of my lines, caluclate the middle point and the differenct between the middle point and middle of screen multiplied my "meters in pixel by x" is a desired distance from center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in visualise method of line function, there is and example

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [ling to project videl](./project_video.mp4), to challenge video [ling to project videl](./project_video.mp4) and to harder challenge video [ling to project videl](./project_video.mp4)

My code works well on project and challenge, but fails on harder challenge. I'll give few points what have to be improved in discussion.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
