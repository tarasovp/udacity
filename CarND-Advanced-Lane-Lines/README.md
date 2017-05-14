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

[undistorted1]: ./output_images/distortion.png "Undistorted"
[undistorted2]: ./output_images/undistorted_track.png "Road Transformed"

[bin1]: ./output_images/bin_5.png "Binary Example 1"
[bin2]: ./output_images/bin_6.png "Binary Example 2"

[projection1]: ./output_images/projected.png "Warp Example"

[vis1]: ./output_images/visualized_2.png "Fit Visual"
[vis2]: ./output_images/visualized_3.png "Fit Visual"


[final]: ./output_images/sample.png "Output"
[video1]: ./project_anotated.mp4 "Project video"


[error]: ./errors/387.png "Mistake example"


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

![Example of unfistroeted images][undistorted1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistorted2]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I've used perspective transform before binaryzation - in my experiments it works better in such way. Unfortanutely I've not seen that sample src and dst are proveided, so I've used my own hardcoded points


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 525, 500      | 250, 250        | 
| 765, 500      | 1030, 250      |
| 1045, 680     | 1030, 690      |
| 260, 680      | 250, 690        |



The code for my perspective transform includes a function called `pr()`, which appears in the 8-10rd code cells of the IPython notebook).  The `pr()` function takes as inputs an image (`img`), and uses global (`src`) and destination (`dst`) variables.  

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Projection example][projection1]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've a littlebit modified functions from udacity course, made them to accept few more parametrs, like, for example on which colorspace ('h' - is H from hls, 'A' is A from LAB and 'gray' is just gray:)) to apply them. This gived me a change to do some experiments and to introduce three functions with different threshholds. In first function I use additionally white and yellow points detection from first project. There is an example, how all three functions work:

![Binary example1][bin1]
![Binary example2][bin2]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I've tried both convolution and sliding window approaches but stopped on sliding window. Of coarse, it's better to implement both and have a choice - if first one not working use the second. Also, I've hardcoded the place where lane's should start - it work fine for project and challenge video, but it will not work for harder challenge at all.

![Visualisation1][vis1]
![Visualisation2][vis2]

Also in a pipeline I used:

* When the lane is detected I just look for points adjasting to the previously detected lines and use them to fit polynom. 
* If I've missed 5+ imagess in a row I'm starting from scratch. 
* I've used 5 images smoothing for provide better visualisation of polynom
* I used "check line" function (see in line class) for check is my line correct. If I've not found lines in the frame I draw blue color on track (using last good fit), and if have not found it for 5+ images in a row than red one
* I used 4 diffetent binatytation functions. If my code misses to find lines for the first function, it tries to use the second. Actually it's possible to add many of functions - but than I've to use better "check line" function, since founded line with a big chance will be incorrect

For mor detailed information please see line class in explore.ipynb.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

It's done in line class. For curvation I just use the same idea as in Udacity course lectures, for the mid point I calculate left and right position of my lines, caluclate the middle point and the differenct between the middle point and middle of screen multiplied my "meters in pixel by x" is a desired distance from center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in visualise method of line function, there is and example

![Example of visualization][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [ling to project video](./project_anotated.mp4), to challenge video [ling to project videl](./challenge_anotated.mp4) and to harder challenge video [ling to project videl](./harder_challenge_anotated.mp4)

My code works well on project and challenge, but fails on harder challenge. I'll give few points what have to be improved in discussion.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This is an example of mistake from the last video

![error][error]

This image contaith both problems of my code:
1) Defenitely needed better binarization. In some images with dark parts or highlights my code misses. This could be done by:
* writing different functions (like I've done+some more) *
* by fitting their parametres using grid search. We can measure when lines are found (== curvation is near the same, posigion is ok, different functions gives the same line on the image - if so, the line is found)
* use other color spaces (I've started with LAB, but did not used it in my project). Maybe to blend different channels
* use some other CV methods (like in first project) or add some ML methods (like clusterization and removing noise using it)

2) The second problem - I've hardcoded that all lines are starting from the bottom of the image. It's not true - on the image there are not right line and the left one starts on the left side. We can:
* If we found one line and sure in it, and we know that it looks like the line is in the right position (according to previous frame) and the second line is not found (not on image or bad quality) we can just calculate is't approximate position
* Better check of the lines. I do not use previous line positions (only in the search of the next one, not in correction check) but it's needed.
