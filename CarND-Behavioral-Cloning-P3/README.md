**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./images/fitting.png "Visualization"
[image2]: ./images/model.png "Visualization"




#1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#Model Architecture and Training Strategy

#1. An appropriate model architecture has been employed

Finally the best architecture was taken from https://github.com/Valtgun/  see selection of model below


#2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (car_model.py lines 213,220,227). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (see notebook or next section). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and few rides in most complicated places (bridge, left turn after bridge and sharp turns on track2)

For details about how I created the training data, see the next section. 

#Model Architecture and Training Strategy

#1. Solution Design Approach

I've tried 4 different network types:
* Modified lenet 
* Model from nvidia (see original nvida papaer - I've just croped images before using their model)
* Tried to use transfer learning and retrain last layers of vgg16 network
* Model based on vgg idea from https://github.com/Valtgun/ 

The performase was


![Performance on different models][image1]

I've trained all of them on the same train dataset and tested on the same data. Model from Valtgun showed the best performance. But I've faced in issue: validation and test errors drops until 25-th epoch, but car stops driving aroud after 10-th epoch! It seems that model starts overfitting - so it's better to found some other test dataset, for example record several mode laps around.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#2. Final Model Architecture

The final model architecture (car_model.py lines 200-240) have the following achitecture:
   
![Architecture][image2]

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


#3. Creation of the Training Set & Training Process

* For each track 4 laps in each direction
* 2 laps with "correction from the sides" for both laps and for both directions
* few rides near most complicated place (bridge, turn after the bridge, sharp turns on track2)


After the collection process, I had 551232 number of data points. I then preprocessed this data by cropping sky and car from the image and taking only 1/4 of pixels.

I finally randomly shuffled the data set and put 1% of the data into a validation set and 1% to the test set. To the trained dataset I've added flipped images.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs accroding to validation loss was 25 - but after 10th epochs model started to overfit. So, it's needed to found other way to find validation dataset. I used an adam optimizer so that manually training the learning rate wasn't necessary.
