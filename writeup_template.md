# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description.

My pipeline consisted of 7-8 steps:
1. I first started by applying a gaussian blur to the image.
2. I then proceeded to convert the image to gray scale.
3. From there I then used Canny edge detection to find the edges in the image.
4. Afterwards I proceeded to mask out a trapezoidal shaped region of interest
5. From the masked region I applied a hough transform. 
   * I modified the function in order for it to return the computed lines as well.
6. From the computes lines I calculated the slopes and proceeded to remove all non finite (NaN, Inf.) slopes and also remove the lines that did not fit within a certain threshold
7. From the slope's direction I could identify if the lines was from the left or right so I separated the slopes from left to right and then averaged the categorized values. I also calculated the intercepts in the same time.
   * From there on I either drew the lines on the image or I returned them so that the video processor could utilise that information.
8. The video processor uses the extra step of implementing a running average of the values in order to smooth the jittering in the video.

In the end I did not use the implemented draw_lines() function but rather used my own version, which had as parameters, the input image, the slope, the intercept and the xs

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
