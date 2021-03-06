# Advanced Lane Finding Project

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

[image1]: ./camera_cal/calibration3.jpg "Original"
[image11]: ./output_images/calibration3.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Original"
[image22]: ./output_images/undist_test1.jpg "Undistorted"
[image3]: ./output_images/straight_lines1.jpg "Straight lines Example 1"
[image33]: ./output_images/straight_lines2.jpg "Straight lines Example 2"
[image4]: ./output_images/tag_straight_lines1.jpg "Unwarp Example"
[image44]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/bin_warp_fit_test1.jpg "Fit Visual"
[image6]: ./output_images/mapped_test3.jpg "Output"
[video1]: ./output_images/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### 2. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image11]

### 3. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]
![alt text][image22]

### 4. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps  in `binary_image.py`) with solid lane lines.  Here's examples of my output for this step. The two straight-lines images are selected from `test_images`.

![alt text][image3]
![alt text][image33]

### 5. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which in the file `perspect_transform.py`. The `warper()` function takes input of an image (`img`). I chose the hardcode the source and destination points in the following manner:

```
	src = np.float32(
		[[img_size[1] / 2 - 63, img_size[0] / 2 + 100],
		[img_size[1] / 2 + 63, img_size[0] / 2 + 100],
		[img_size[1] / 2 + 485, img_size[0]],
		[img_size[1] / 2 - 450, img_size[0]]])

	dst = np.float32(
    	[[(img_size[1] / 4), 0],
    	[(img_size[1] * 3 / 4), 0],
    	[(img_size[1] * 3 / 4), img_size[0]],
    	[(img_size[1] / 4), img_size[0]]])
```
This resulted in the following source and destination points:

src: [[  577.   460.]
 [  703.   460.]
 [ 1125.   720.]
 [  185.   720.]]

dst: [[ 320.    0.]
 [ 960.    0.]
 [ 960.  720.]
 [ 320.  720.]]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image44]

### 6. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The file `find_lines.py` contains the code for finding lane-line pixels. The overall procedure I took is first establishing historgram of the bottom half of the binarized-warped image. By using the historgram, I can find the two peaks which are possible left/right lanes. By using sliding window mechanism, I can aggregate the nonzero pixels slide by slide and form my left/right lanes. Then fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

### 7. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 95 through 122 in my code in `find_lines.py`

### 8. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 11 through 45 in my code in `map_lane.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

### 9. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

### 11. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that my original pipeline failed at the last tree shadow part. I think it mistakenly detects the shaow as right lane, but it is a shadow edge toward left in fact. As a result, I implemented a filter machanism which takes corrlation between frames into account to filter out strange curves and a smoother machanism which uses several curve infomation to form an average curve. The result shows that these mechanisms did help me in this scenario.

