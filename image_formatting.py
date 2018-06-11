import cv2
import glob

def age_group(age):
	if age < 2:
		return 1
	elif age < 10:
		return 2
	elif age < 16:
		return 3
	elif age < 28:
		return 4
	elif age < 51:
		return 5
	elif age < 75:
		return 6
	else:
		return 7


path = './UTKFace/*.jpg'    
files = glob.glob(path) 
images = []
count_list = [241, 241, 241, 241, 241, 241, 241]

for name in files: 
	print (name) 
	image = cv2.imread(name, 0)
	pre = name.split("_")[0]
	age = pre.split("/")[-1]
	group = age_group(int(age))
	new_path = "./faces2/face_" + str(group).zfill(3) + "_" + str(count_list[group-1]).zfill(5) + ".png"
	count_list[group-1] += 1
	cv2.imwrite(new_path,image)

