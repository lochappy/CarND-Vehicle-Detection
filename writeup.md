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
[image1]: ./Figures/car_notcar.png
[image2]: ./Figures/hog_sample.png
[image3]: ./Figures/sliding_windows.png
[image4]: ./Figures/sliding_windows_results.png
[image5]: ./Figures/bboxes_and_heat.png
[image6]: ./Figures/label_and_final.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the `VehicleDetection.ipynb` IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images in the 2nd code cell of the IPython notebook. There are totally 8792 vehicle and 8968 non-vehicle images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed a random image from the `vehicle` classe and displayed them to get a feel for what the `skimage.hog()` output looks like in the 3rd code cell of the IPython notebook.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Specifically, for each set of parameters, I extracted the feature (in 4th code cell ), then trained an `SVM` on them (in 5th code cell), and finally recorded the accuracy on the validation set. The final set of parameters with the best accuracy landed at:

`color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code of this part is in the 5th code cell of the IPython Notebook. Specifically:
* I first normalized the extracted features using `sklearn.preprocessing.StandardScaler()`.
* Then, I split the data into train/test set by using `sklearn.model_selection.train_test_split()`.
* Next, I trained a linear SVM on the train set.
* Finally, I evaluated the trained SVM on test set and logged out the accuracy.

The SVM having the best accuracy on the test set was chosen.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The implementation of this part could be found in `find_cars()` function of the 6th code cell of the IPython notebook. In flollowing steps:
* Input image is converted to appropriate color space.
* The converted image is then resized to an appropriate size specified by the `scale` parameter.
* HOG features of each channel of the resized image are then extracted.
* Finally, the sliding window approach was applied in the following nested loop.

`for xb in range(nxsteps):
     for yb in range(nysteps):
         ...
`

Here is an example of detected sliding windows.

![alt text][image3]

After many trial-error experiments, I found that the `scales=[1.2, 1.35, 1.5]` gave the appropriate results at a very resonable false positives and processing speed.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using `YUV` 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  In order to optimize the performance of my classifier, I have tried various range of `[min_scale, max_scale]` searching scale as well as the number of scales `num_scales` in each range. Finally, the range `[1.2, 1.5]` with `num_scales=3` gave the best result. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./detected_vehicle_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The implementation of this part could be found in the 7th code cell of the IPython notebook. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the final resulting bounding boxes.
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

a. Processing speed is quite slow, saying that the current method runs at 1.05 FPS. There could be some room for improvement such as:
 * Doing the sliding window in parallel by using multiple threads, instead of the current single.
 * Restrict the HOG extraction to a smaller region of interest, which could be initialized by prvious detection.
 * Although linear SVM is fast, it is still not fast enough to run at realtime. We can replace this by a faster method like cascade of decision trees.
 * etc

b. The pipeline will probably fail at some harsh conditions such as rain, night, etc...

c. Detections are not stable, saying that the bounding boxes are sometimes shunk to much smaller than the  real object size. The problem could be at the detecting scale and sliding steps.

d. Object detection using sliding window approach is quite outdated. The one with deep learning approach should give more accurate and stable result, such as SSD, YOLO,...
