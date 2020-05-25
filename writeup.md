# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./report_images/step1.png "Gaussian"
[image2]: ./report_images/step2.png "Grayscale"
[image3]: ./report_images/step3.png "Canny"
[image4]: ./report_images/step4.png "Mask"
[image5]: ./report_images/step5.png "Hough"
[image6]: ./report_images/step6.png "Lines"

---

### Reflection

### 1. Describe your pipeline. As part of the description.

My pipeline consisted of 7-8 steps:
1. I first started by applying a gaussian blur to the image.
![alt text][image1]
2. I then proceeded to convert the image to gray scale.
![alt text][image2]
3. From there I then used Canny edge detection to find the edges in the image.
![alt text][image3]
4. Afterwards I proceeded to mask out a trapezoidal shaped region of interest
![alt text][image4]
5. From the masked region I applied a hough transform. 
   * I modified the function in order for it to return the computed lines as well.
![alt text][image5]
6. From the computes lines I calculated the slopes and proceeded to remove all non finite (NaN, Inf.) slopes and also remove the lines that did not fit within a certain threshold
7. From the slope's direction I could identify if the lines was from the left or right so I separated the slopes from left to right and then averaged the categorized values. I also calculated the intercepts in the same time.
   * From there on I either drew the lines on the image or I returned them so that the video processor could utilise that information.
![alt text][image6]
8. The video processor uses the extra step of implementing a running average of the values in order to smooth the jittering in the video.

In the end I did not use the implemented draw_lines() function but rather used my own version, which had as parameters, the input image, the slope, the intercept and the xs.

### 2. Identify potential shortcomings with your current pipeline

One shortcoming of the pipeline is the orientation of the camera, if the camera is orientated too upwards or to the side, the lines might get filtered out either by teh angle threshold that I have implemented or by the region masking where the line is cut off
Another shortcoming of this pipeline is if there is a lot of light or too little light then there will be issues in finding the edges in the image thus causing the pipeline to miss the lanes

### 3. Suggest possible improvements to your pipeline

A possible improvement to my pipelines would be to be able to accommodate for the change in perspective of the camera in order to standardize the line orientation.
