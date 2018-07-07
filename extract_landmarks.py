import cv2
import glob
import numpy as np
import imutils 

def extract_landmarks():
	path = './faces/*.png'    
	files = glob.glob(path) 
	number_of_features = end - start + 1

	lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
	har_params = ()
	gab1_params = ()
	gab2_params = ()

	count = 0
	bar_len = 60
	total = number_of_features * 7
	for name in files: 
		img = Image(name)
		if (img.number >= start and img.number <= end):
			image = cv2.imread(name, 0)
			### LANDMARK METHOD ###
			landmarks = []
			lib_pat.progress(count, total, name)
			### LEFT EYEBROW
			landmark = get_landmark(image, landmarks, "left eyebrow", slack = 0.15)
			extraction_routine("./eyebrowL/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params, gab2_params)
			### RIGHT EYEBROW
			landmark = get_landmark(image, landmarks, "right eyebrow", slack = 0.15)
			extraction_routine("./eyebrowR/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params, gab2_params)
			### NOSE
			landmark = get_landmark(image, landmarks, "nose", slack = 0.1)
			extraction_routine("./nose/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params, gab2_params)
			### LEFT EYE
			landmark = get_landmark(image, landmarks, "left eye", slack = 0.1)
			extraction_routine("./eyeL/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params, gab2_params)
			### RIGHT EYE
			landmark = get_landmark(image, landmarks, "right eye", slack = 0.1)
			extraction_routine("./eyeL/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params, gab2_params)
			### MOUTH
			landmark = get_landmark(image, landmarks, "mouth", slack = 0.1)
			extraction_routine("./mouth/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[landmark], lbp_params[0], lbp_params[1], har_params, har_params, gab1_params, gab2_params)
			count += 1
	print("")

def get_landmark(image, landmarks, part, slack = 0):
	if (part == "left eyebrow"):
		rango = range(17,22)
	elif (part == "right eyebrow"):
		rango = range(22,27)
	elif (part == "nose"):
		rango = range(27,36)
	elif (part == "left eye"):
		rango = range(36,42)
	elif (part == "right eye"):
		rango = range(42,48)
	elif (part == "mouth"):
		rango = range(48,68)

	x_min = [10000,10000]
	x_max = [0,0]
	y_min = [10000,10000]
	y_max = [0,0]
	landmark = image.copy()

	for i in rango:
		landmark[landmarks[i][1]][landmarks[i][0]] = 0
		if (landmarks[i][0] > x_max[0]):
			x_max = landmarks[i]
		if (landmarks[i][0] < x_min[0]):
			x_min = landmarks[i]
		if (landmarks[i][1] > y_max[1]):
			y_max = landmarks[i]
		if (landmarks[i][1] < y_min[1]):
			y_min = landmarks[i]
	x_slack = int((x_max[0] - x_min[0]) * slack)
	y_slack = int((y_max[1] - y_min[1]) * slack)
	landmark = image[y_min[1] - y_slack:y_max[1] + y_slack, x_min[0] - x_slack:x_max[0] + x_slack]
	# cv2.imshow("Image", landmark)
	# cv2.waitKey(15000)
	# cv2.waitKey(0)



extract_landmarks()


# landmarks = np.load('landmarks0.npy')
# image = cv2.imread('me1.jpg', 0)
# image = imutils.resize(image, width=500)
# # cv2.imshow( "Image", image)
# # cv2.waitKey(0)
# slack = 0.1
# get_landmark(image, landmarks, "left eyebrow", slack = slack)
# get_landmark(image, landmarks, "right eyebrow", slack = slack)
# get_landmark(image, landmarks, "nose", slack = slack)
# get_landmark(image, landmarks, "left eye", slack = slack)
# get_landmark(image, landmarks, "right eye", slack = slack)
# get_landmark(image, landmarks, "mouth", slack = slack)
# # cv2.imshow( "Image", image)
# # cv2.waitKey(0)




