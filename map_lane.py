import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from perspect_transform import warper
from binary_image import binary_pipeline
from find_lanes import find_lane,find_curvature,find_center,fast_find_lane
from camera_cal import cal_undistort

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterationsre
        self.bestx = np.zeros(720)     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.zeros(3)  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.diffs = np.array([0,0,0], dtype='float') 

    def append_fit(self, fit, fitx):
        '''
        Filter and smoother algorithm to filter out strange curves and smooth the curve. 
        '''
        iter_num = 5
        thres = 0.75
        if len(self.recent_xfitted) == 0:
            self.current_fit.append(fit)
            self.recent_xfitted.append(fitx)
        else:
            # By using correlation, filter out curves below a tuned threshold
            if np.corrcoef(np.vstack((self.best_fit,fit)))[0,1] > thres:
                self.current_fit.append(fit)
                self.recent_xfitted.append(fitx)
            else:
                if len(self.recent_xfitted) < iter_num:
                    self.detected = False

            if len(self.recent_xfitted) >= iter_num:
                self.current_fit.pop(0)
                self.recent_xfitted.pop(0)
                self.detected = True

        # Smooth by averaging several curves.
        self.bestx = np.average(self.recent_xfitted, axis=0)
        self.best_fit = np.average(self.current_fit, axis=0)

left_lanes = Line()
right_lanes = Line()

def map_lane(img):
    combined = binary_pipeline(img)
    combined *= 255

    undist,warped,Minv = warper(combined)

    if not left_lanes.detected or not right_lanes.detected:
        _, ploty,left_fit,right_fit,left_fitx,right_fitx = find_lane(warped)
    else:
        _, ploty,left_fit,right_fit,left_fitx,right_fitx = fast_find_lane(warped, left_lanes.best_fit, right_lanes.best_fit)

    left_lanes.append_fit(left_fit, left_fitx)
    right_lanes.append_fit(right_fit,right_fitx)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lanes.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lanes.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    undist = cal_undistort(img)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_curad,right_curad = find_curvature(img.shape[0],ploty,left_lanes.bestx,right_lanes.bestx)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Left Curad: ' + str(np.round(left_curad,4)) + 'm Right Curad: ' + str(np.round(right_curad,4)) + 'm',(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    offset = find_center(left_lanes.bestx, right_lanes.bestx)
    cv2.putText(result,'Offset from center: ' + str(np.round(offset,4)) + 'm',(50,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane finding pipeline.')
    parser.add_argument('prefix', nargs='?', type=str, default='', help='File path to the image.')

    args = parser.parse_args()

    print('File: ' + args.prefix)
    img = cv2.imread(args.prefix)

    result = map_lane(img)

    cv2.imwrite('./output_images/mapped_' + args.prefix.split('/')[-1], result)