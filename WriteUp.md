
# **Vehicle Detection Project**
---
## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README
The solution to this problem is divided into ***Three*** parts.

> A. Extracting features from the input image

> B. Selecting suitable classifier

> C. Implementing `sliding window` technique for identify the object

## A. Extracting Features
To get the correct prediction and to generalise our model, we have to extract the features specific to the problem statement. The following three features are used in this project
> I. Spatial Bins

> II. Color histogram

> III. Histogram of Oriented Gradients

### I. Spatial Bins
A seperate functon `bin_spatial` is written to extract the Spatial vector from the input image. (cell no: 5 of the IPython notebook)
> Transformation of input image
>> ![alt text](.\examples\Spatial_bins.jpg)

> Feature vector
>> ![alt text](.\examples\Spatial_Features.JPG)


### II. Color Histogram
Function `color_hist` is used for retriving color histogram for the given image. (cell no: 7 of the IPython notebook)
> Feature vector
>> ![alt text](.\examples\Color_Hist_Feature.JPG)

### III. Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code for this step is contained in the cell no : 9 of the IPython notebook (or in lines 2 through 21).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text](.\examples\car_not_car.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

> Transformation of input image
>> ![alt text](.\examples\HOG_Transform.JPG)
> Feature Vector
>> ![alt text](.\examples\HOG_Feature.JPG)

#### 2. Explain how you settled on your final choice of HOG parameters.

I have mainly changed the color channel in the HOG transform. Initially, I have used the "RGB" color space, even though it gave good accuracy in test set. It doesn't work well in the test video. So, I have changed the color space to "YUV". Instead of adding all channels in HOG, I have used only `Y-Channel` which has large effect on the output.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have combined spatial feature, color histogram and HOG feature. Each feature has different range of value. So, the model/classifier could be biased on some particular features. In order to avoid this effect, I have normalized my input feature vector.

> From the below image, we can see the output range is reduced to -1 to 5
>> ![alt text](.\examples\Normalization.JPG)

Using the above data, I have trained ***Two*** classifiers.
* SVC (cell No : 20)
> * linear
> * rbf
* Random forest classifier (cell No : 21)

Initially `RandomForestClassifier` are trained fastly and gave good accuracy without changing any parameters. But `SVC` classifier with `kernal=rbf, C=10` fits my data with ***99.35%*** accuracy.

### C. Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window is present in cell no 24.Before constructing the sliding window, I fixed the boundary for sliding window as follows,
```python
X_Range=(int(test_img.shape[1]*0.45),test_img.shape[1])
Y_Range=(int(test_img.shape[0]*0.55),int(test_img.shape[0]*0.96))
```
The above code will eliminate upper & left half of my input image. So, i can able to increase my computation speed.

I chose window size as (96,96) which is the average value to detect any car in that region. And overlapping of 70%, gives more boundary boxes on the car. This helps in reducing the false positive.

![alt text](.\examples\SlidingWindowBoundary.JPG)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I will explain the pipeline through my code.

***Pipeline : *** `update_frame` (cell no :40)

> `search_engine` : (cell no : 26)
>> * Fixing boundary line for search window
>> * In each window, extract feature and predict using classifier
>> * For car images, return the co-ordinates

> `add_heat` : (cell no : 28)
>> * Create a heat image according to the boundary boxes provided by `search_engine`

> `apply_threshold` : (cell no:30)
>> * Apply the specified thershold to differentiate the object as well as reduce the false positive.

> `draw_labeled_bboxes` : (cell no:32)
>> * This will take a copy of the input image & draw the boxes identified after thresholding

***Final Result For Single Frame***

![alt text](.\examples\PipeLine.JPG)

***Optimization :***
The main problem with my classifier is detecting false positives. To reduce it, I followed ** Two ** methods. They are,
1. Resizing the boundary line
2. Introduced averageing technique
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](.test_videos_output/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here are six frames and their corresponding heatmaps:
![alt text](.\examples\SampleFrame1.JPG)
![alt text](.\examples\SampleFrame2.JPG)
![alt text](.\examples\SampleFrame3.JPG)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The main problem which I faced was selecting the classifier and reducing the false positives. There were no short cuts to identify the classifier from its accuracy value. We have to run the video to check whether it is working or not.

This may fail in new environment. Instead of SVM, if we use CNN, We can generalize our model for other new environment also
