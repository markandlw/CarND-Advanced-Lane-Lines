import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from perspect_transform import warper
from binary_image import binary_pipeline
from find_lanes import find_lane,find_curvature,find_center
from camera_cal import cal_undistort

def map_lane(img):
	combined = binary_pipeline(img)
	combined *= 255

	undist,warped,Minv = warper(combined)

	out_img,ploty,left_fit,right_fit,left_fitx,right_fitx = find_lane(warped)

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	undist = cal_undistort(img)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	left_curad,right_curad = find_curvature(img.shape[0],ploty,left_fitx,right_fitx)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result,'Left Curad: ' + str(np.round(left_curad,4)) + 'm Right Curad: ' + str(np.round(right_curad,4)) + 'm',(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)

	offset = find_center(left_fitx, right_fitx)
	cv2.putText(result,'Offset from center: ' + str(np.round(offset,4)) + 'm',(50,100), font, 1,(255,255,255),2,cv2.LINE_AA)

	return result

parser = argparse.ArgumentParser(description='Lane finding pipeline.')
parser.add_argument('prefix', nargs='?', type=str, default='', help='File path to the image.')

args = parser.parse_args()

print('File: ' + args.prefix)
img = cv2.imread(args.prefix)

result = map_lane(img)

cv2.imwrite('./output_images/mapped_' + args.prefix.split('/')[-1], result)