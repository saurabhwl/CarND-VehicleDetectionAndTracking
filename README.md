**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./results/overview_training_images.png
[image2]: ./results/overview_features.png
[image3]: ./results/sliding_windows.png
[image4]: ./results/overview_still_images.png
[image5]: ./results/overview_video_frames.png
[image6]: ./results/labels_map.png
[image7]: ./results/output_bboxes.jpg
[video1]: ./results/project_video.mp4

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The implementation can be found in of the file `get_hog_features()` from the file `Features.py`.
In the `VehicleDetection` class a wrapper method `__get_hog_features()` exists to calculate the features with the current settings specified in the configuration file.

The features are either returned as a 1D vector or as a matrix.
Later is helpful when applying the sliding windows as we do not need to calculate the hog features for each window but once for the whole image.

A visualization of the HOG features shows that the orientation of the gradient of the single cells give quite a lot of vertical and horizontal lines. The shape of the car can still be estimated.
For non-vehicle images such a shape is not identifiable.
![alt text][image2]

The image also shows the histogram feature for the HLS color space.

####2. Explain how you settled on your final choice of HOG parameters.

I tried several combinations of orientation, pixel per cell and cells per block.
After visualizing the results I trained the classifier and checked the results on the video.

In the end I choose parameters close to those mentioned in the lectures.

```
Orientation: 9
PixelPerCell: 8
CellPerBlock: 2
```

Lowering the number of orientations gave worse results while increasing the number did not really boost the performance.

The hog features where extracted for all three channels of the image after transforming it to another color space (I used YCrCb in the end.)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training of the classifier takes place in the `__train_classifier()` method of the
`VehicleDetection` class.

It first reads the dataset. You can choose wether to read the combined GTI, KITTI and project dataset or the Udacity released labeled dataset.
This image shows an example of the vehicle and non-vehicle class.

![alt text][image1]

Afterwards this data is shuffled and split into a training and test set.
The test set was chosen with 10% and is relatively small as the test set is not completely independent.
The dataset contains many consecutive frames and additional work would be necessary to
have a dataset with independent images and therefore a real quality measurement.

From training and test set we extract HOG, color histogram and spatial features.
The data is normalized and deals as input for a linear Support Vector Classifier.

The results are stored and can be re-used later on.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The final implantation of the sliding window search can be found in the `__detect_vehicles_frame()` method of the `VehicleDetection` class.

In this implantation we do not set the explicit window size but work with a 64x64 window all of time and rescale the image. We use a scaling of 1.5, 1.25 and 1.0 what corresponds to 96x96, 80x80 and 64x64 search windows.
The overlap was chosen to be 0.25. This means that it takes 4 windows to make a complete "step".
Further the search region was limited to an area relevant for vehicles starting at height 400px and going to 656px.

This is visualized in the image below with 96x96 windows shown in green and 64x64 are annotated in blue.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline works quite well on still images.
It detects the vehicles in the images on your lane and even is capable of detecting images on the other traffic lanes.
Some false positives occur but their number is rather low.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./results/project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

First all windows, for which the classifier returns "vehicle" are summed up in a heatmap.

![alt text][image5]

A real vehicle shows multiple detections for different sized sliding windows.
That's why the heatmap is thresholded and only areas with multiple detections are considered further on.

The `scipy.ndimage.measurements.label()` is used to cluster the heatmap into single "vehicles". It labels each enclosed heatmap area into a separate cluster which is indicated by the different brightness levels.

![alt text][image6]

In the end these results are taken to create the annotated image:
![alt text][image7]

To improve stability the results of several consecutive frames are combined to a single heatmap. This is more or less low-pass filter.

The code below shows this approach:
```
# Find currently detected vehicles
current_estimation = labels[0] > 0
current_estimation = current_estimation.astype(np.int)

if not self.tracking_heatmap_init:
    # Init tracking algorithm
    self.tracking_heatmap_init = True
    self.tracking_heatmap = current_estimation * 3
    result_img = img
else:
    # Decrease all elements by one (loose confidence)
    self.tracking_heatmap -= 1
    # Increase tracking heatmap for current detections
    self.tracking_heatmap = np.clip(self.tracking_heatmap + (current_estimation * 4), 0, 10)

    # Threshold tracking heatmap and draw labels
    tr_heatmap = apply_threshold(np.copy(self.tracking_heatmap), 7)
    labels = label(tr_heatmap)
    result_img = draw_labeled_box(img, labels)
```
This code can be found in the `__detect_vehicles_video()` method.

The `current_estimation` is derived from the heatmap labeled image shown below.
Wherever a vehicle is detected it has 1, anywhere else the value is 0.

For the first cycle the tracking is initialized and the `current_estimation` is multiplied with 3 and stored in the internal variable `tracking_heatmap`
For all consecutive frames the `tracking_heatmap` is decreased by 1 for all pixels where no vehicle was detected and increased by 3 for detections.
The results are saturated to a range between 0 und 10.
These parameters where chosen heuristically.

Whenever a pixel has a value bigger than 7 it is detected as a vehicle and annotated in the image.
This means that at least 3 detections in consecutive frames are necessary to initially detect a vehicle.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Incompatible image data types caused problems at the beginning.
The training images where provided in png and read in floating point representation while the
test images and video frames hat a UInt8 representation. This showed that carefully looking
at the data before applying machine learning approaches is very important.
- Calculating the hog features for the whole image and not for the search window not only improved
speed but improved the results as the cut-out images showed a lot of discontinuities.
- Comparing results is complicated if you don't have a ground truth available. Optimizing the parameters can
only be done by viewing and manually checking the results. This shows the need for tons of labeled data.
- The detection of vehicles in the image using a SVC is not quite stable. It takes a lot of effort to overcome
the shortcomings like heatmap and
- In the future the results of the sequential frames should not only get low passed but predicted using e.g. a Kalman filter to improve the results. Transforming the bounding boxes to a birds eye view (perspective transform as in lane finding project) makes the tracking and the results independent of image distortions.
