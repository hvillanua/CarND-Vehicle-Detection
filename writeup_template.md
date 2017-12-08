## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/roi.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/pca_variance.png
[image4]: ./output_images/scale1.png
[image5]: ./output_images/scale1.5.png
[image6]: ./output_images/scale2.png
[image7]: ./output_images/all_scales.png
[image8]: ./output_images/heatmap.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/hvillanua/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 29 through 46 of the file called `helper.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YUV` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters while training the SVM and after finishing the pipeline to get a feel of what works best.
I also asked for suggestion on Slack and used those to get my final set of hyperparameters:

```
# HOG + color hist + binned color features
spatial = 16
histbin = 16
orient = 12
pix_per_cell = 16
cell_per_block = 2
cspace = 'YUV'
hog_channel = 'ALL'

# PCA
n_components = 150

# SVM
kernel='rbf'
C=10
gamma=0.005
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the notebook cells 3 and 4 I get all the features from car and non-car images. Then I opted to use PCA to reduce the dimensionality
of the input to the SVM classiffier. This would also let me use a more complex kernel since the classifier predictions would be much faster
with 150 pca reduced features than the original 4000+ features.
When the pipeline was finished, I tried both linear and rbf kernel. Rbf outperformed linear kernel by avoiding most false positives
while keeping the true positives high.

Here is the explained variance from the features extracted by the PCA:

![alt text][image3]

PCA fit done in 4.249s; Total explained variance: 0.744602646594

8.23 Seconds to train SVC...; Test Accuracy of SVC =  0.9989; 0.00735 Seconds to predict 10 labels with SVC

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided on 3 region of interests that would be used for different sliding window scales.
Red is for scale 1, green for 1.5, blue for 2:

![alt text][image1]

The choice was made for improved speed at the cost of less windows matching over the image. Scales above 2 seemed unnecesary as they would classify the same or less than scale 2.
Scales under 1 took a lot of time to compute and were prone to false positives.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image8]

I also implemented two global variables to keep track of the last 10 frames. This proved to make the bounding boxes more robust, as well as allowing the filtering
of false positives.
Instead of filtering each frame, I would just add them to the pool of last 10 heatmaps, add them and filter them with a value of 6. This means that even small disturbances appearing in at least
5 of the frames would always be ignored. On the other hand it takes a bit longer for new objects to appear on the heatmap if it doesn't have many sliding windows identifiying it.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of my time was spent tuning the hyperparameters and choosing the right features. I wanted a good pipeline that barely gets any false positives, while still being
as fast as it can. This led me to use all 3 different techniques explained above and the rbf kernel for accuracy, as well as using PCA and increasing the cell_size to (16, 16) for faster training.
The algorithm is still lacking a lot in speed. It wouldn't be useful for real-time application. It would also probably fail if there is a high concentration of cars,
extreme lightning conditions or bad weather.
All of this said, I'm happy that the accuracy is pretty good. As a next step it would be a good idea to try a CNN or some kind of preprocessing to limit the region of interest
for each sliding window to apply to.
