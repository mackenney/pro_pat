import cv2
import glob

path = './faces/*.png'    
files = glob.glob(path) 
images = []
# count_list = [241, 241, 241, 241, 241, 241, 241]

for name in files: 
	group = int(name[-13:-10])
	number = int(name[-9:-4])
	if (number <= 1006):
		# print (name)	 
		image = cv2.imread(name, 0)
		 
		vertical_image = image.copy()
		vertical_image = cv2.flip(image, 1)
		new_path = "./faces/face_" + str(group).zfill(3) + "_" + str(number + 1006).zfill(5) + ".png"
		print(new_path)
		# cv2.imshow( "Original", image )
		# cv2.imshow( "Vertical flip", vertical_image )
		# cv2.waitKey(0)
		cv2.imwrite(new_path,vertical_image)

