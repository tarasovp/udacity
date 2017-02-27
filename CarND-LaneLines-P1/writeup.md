#**Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report



---

### Reflection

###1. Builded pipeline:

My pipeline consisted of 9-10 steps.

* I filled everything but a trapeze in the center of the image to black. Not a triangle to cut unneeded detailes in the upper edge 

* I've chosen yellow and white colours from the image and fill all other colours with black. Here I've used a trick: if picture is lights used bigger treshholds

* I've converted image to greyscale

* I've used Gaussian Blur to get rid of unneeded detailes

* I've used Canny to calculate gradient 

* I've truncated with a 3-pixel smaller trapeze to get rid of gradient on the edge of first trapeze

* I've applied hough lines to get lines from the image

* For getting segments I've filtered lines which have for sure wrong angle or their bottom not near the "wheels"

* For getting solid lines the same, but averaged angle, bottom part of the line and calculated the top as the angle*height+bottom

* Finally for solid lines I've applied smoothing (averaged 10 previos lines), since car moves smoothly all rapid changes come from program mistake



###2. Identify potential shortcomings with your current pipeline

* Yellow and white color - can be chosen wrong in some cases

* I've realised that I've to add segments in 5 minutes before this commit. So in some segments I've wrong lines - a little parameter tuning can help to get rid of them


###3. Suggest possible improvements to your pipeline

* We have to choose it according to the current image. The best thing is to use some king of standardisation for the image. I've impemented just very simple case for it: if mean color<130 then use bigger treshholds

* We can make this program as a function with a lot of parameters (actually it is now, but some of them are hardcoded). Than using video where we have already painted the lines we can use different optimisation technics to chose this parameters, it will be much more effective than use it by hand