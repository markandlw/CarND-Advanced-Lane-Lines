import argparse
import cv2
import numpy as np
import camera_cal as cc

def warper(img):
	img_size = img.shape
	print(img_size)

	src = np.float32(
		[[img_size[1] / 2 - 63, img_size[0] / 2 + 100],
		[img_size[1] / 2 + 63, img_size[0] / 2 + 100],
		[img_size[1] / 2 + 485, img_size[0]],
		[img_size[1] / 2 - 450, img_size[0]]])

	print(src)

	dst = np.float32(
    	[[(img_size[1] / 4), 0],
    	[(img_size[1] * 3 / 4), 0],
    	[(img_size[1] * 3 / 4), img_size[0]],
    	[(img_size[1] / 4), img_size[0]]])
    	
	print(dst)

	M = cv2.getPerspectiveTransform(src, dst)

	undist = cc.cal_undistort(img)

	warped = cv2.warpPerspective(undist,M, (undist.shape[1],undist.shape[0]),flags=cv2.INTER_LINEAR)

	pts = np.array(src, np.int32)
	pts = pts.reshape((-1,1,2))
	cv2.polylines(undist,[pts],True,(0,0,255),3)

	pts = np.array(dst, np.int32)
	pts = pts.reshape((-1,1,2))
	cv2.polylines(warped,[pts],True,(0,0,255),3)	

	return undist,warped

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perspective transform output.')
    parser.add_argument('prefix', nargs='?', type=str, default='', help='File path to the image.')

    args = parser.parse_args()

    print('File: ' + args.prefix)
    img = cv2.imread(args.prefix)

    undist,dest = warper(img)

    cv2.imwrite('./output_images/tag_' + args.prefix.split('/')[-1], undist)
    cv2.imwrite('./output_images/warped_' + args.prefix.split('/')[-1], dest)

