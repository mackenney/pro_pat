import librerias_patrones as lib_pat
import main
import cv2
import glob
import numpy as np
import multiprocessing as mp
from sklearn.metrics import confusion_matrix as CM
# import sys

# def lib_pat.progress(count, total, status='', up=True):
#     bar_len = 60
#     filled_len = int(round(bar_len * count / float(total)))

#     percents = round(100.0 * count / float(total), 1)
#     bar = '=' * filled_len + '-' * (bar_len - filled_len)
#     if (not up):
#     	data_on_first_line = '\n'
#     	sys.stdout.write(data_on_first_line)
#     sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
#     sys.stdout.flush()  
#     if (not up):
#     	CURSOR_UP_ONE = '\x1b[1A' 
#     	data_on_first_line = CURSOR_UP_ONE 
#     	sys.stdout.write(data_on_first_line)

class Image:
	def __init__(self, name, npy=False):
		self.group = int(name[-13:-10])
		self.number = int(name[-9:-4])
		# if npy:
		# 	print(name[-13:-10])
		# 	print(name[-9:-4])
		# 	self.group = int(name[-13:-10])
		# 	self.number = int(name[-9:-4])
		# else:
		# 	self.group = int(name[14:17])
		# 	self.number = int(name[18:23])

def extraction_routine(arr_name, images, lbp_grids, lbp_dists, har_grids, har_dists, gab_grids1, gab_grids2):
  """
  Saves to arr_name .npy the array of features extracted.
  Params MUST be tuples, use (1,) for single values
  :param lbp_grids: tuple
  :param lbp_dists: tuple, must be the same size than ibp_grids
  :param har_grids: idem
  :param har_dists: idem
  :param gab_grids1: idem
  :param gab_grids1: idem
  :return: void
  """
  count = 0
  total = len(lbp_grids) + len(har_grids) + len(gab_grids1) + len(gab_grids2)
  lbps = []
  for i in range(len(lbp_grids)):
    # print('LPB Extraction, Iteration {}/{}'.format((i + 1), len(lbp_grids)))
    with mp.Pool() as p:
      lbp = p.starmap(lib_pat.get_LBP, [(images[j], lbp_dists[i], lbp_grids[i], j) for j in range(len(images))])
    lbps.append(lbp)
    count += 1
    lib_pat.progress(count, total, up=False)
  lbp_feats = np.concatenate(lbps, axis=1)

  hars = []
  for i in range(len(har_grids)):
    # print('Haralick Extraction, Iteration {}/{}'.format((i + 1), len(har_grids)))
    with mp.Pool() as p:
      har = p.starmap(lib_pat.get_Haralick, [(images[j], har_dists[i], har_grids[i], j) for j in range(len(images))])
    hars.append(har)
    count += 1
    lib_pat.progress(count, total, up=False)
  har_feats = np.concatenate(hars, axis=1)

  gabs1 = []
  for i in range(len(gab_grids1)):
    # print('Gabor Extraction, Iteration {}/{}'.format((i + 1), len(gab_grids1)))
    with mp.Pool() as p:
      gab = p.starmap(lib_pat.get_Gab, [(images[j], gab_grids1[i], j) for j in range(len(images))])
    gabs1.append(gab)
    count += 1
    lib_pat.progress(count, total, up=False)
  gab_feats1 = np.concatenate(gabs1, axis=1)

  gabs2 = []
  for i in range(len(gab_grids2)):
    # print('Gabor Extraction, Iteration {}/{}'.format((i + 1), len(gab_grids2)))
    with mp.Pool() as p:
      gab = p.starmap(lib_pat.get_Gab_real_im, [(images[j], gab_grids2[i], j) for j in range(len(images))])
    gabs2.append(gab)
    count += 1
    lib_pat.progress(count, total, up=False)
  gab_feats2 = np.concatenate(gabs2, axis=1)

  feats = np.concatenate((lbp_feats, har_feats, gab_feats1, gab_feats2), axis=1)
  np.save(arr_name, feats)
  return feats

def classify(feats, cantidad, iterations, separate_ratio):
	lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
	har_params = ((1, 1, 1, 2, 5), (1, 10, 20, 11, 8))
	gab1_params = (1, 2, 5, 10)
	gab2_params = (1, 2, 5, 10)
	labels = main.generate_labels(lbp_params[0], har_params[0], gab1_params, gab2_params)

	print('Removing features with low variance')
	feats, labels = lib_pat.delete_zero_variance_features(feats, labels, 0.1)

	lda_scores = []
	mlp_scores = []
	for i in range(iterations):
		print('Separating Features...')
		X_tr, X_te, y_tr, y_te = lib_pat.separate_train_test(feats, separate_ratio, cantidad)

		print('Reducing features by transformation')
		# X_tr, X_te = main.reduction_routine(feats, labels, separate_ratio, .99, cantidad)
		print('Final reduction (for no colinear features)')
		X_tr, X_te = lib_pat.dim_red_auto_PCA(X_tr, X_te, ratio=.9)

		print('Classification via LDA solver=svd')
		k1, k1_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te, solver='svd')
		lda_scores.append(k1_score)

		print('Classification via MLP')
		k2, k2_score = lib_pat.classification_LDA(X_tr, X_te, y_tr, y_te)
		mlp_scores.append(k2_score)

		# np.savetxt('k1', CM(y_te, k1), fmt='%2i', delimiter=',')
		# np.savetxt('k2', CM(y_te, k2), fmt='%2i', delimiter=',')
	lda_mean = sum(lda_scores) / float(len(lda_scores))
	print('LDA mean accuracy:', lda_mean)
	mlp_mean = sum(mlp_scores) / float(len(mlp_scores))
	print('MLP mean accuracy:', mlp_mean)


def extract(start, end):
	path = './faces/*.png'    
	files = glob.glob(path) 
	number_of_features = end - start + 1

	lbp_params = ((1, 1, 2, 2, 5), (5, 10, 8, 15, 6))
	har_params = ((1, 1, 1, 2, 5), (1, 10, 20, 11, 8))
	gab1_params = (1, 2, 5, 10)
	gab2_params = (1, 2, 5, 10)

	count = 0
	bar_len = 60
	total = number_of_features * 7
	for name in files: 
		img = Image(name)
		if (img.number >= start and img.number <= end):
			image = cv2.imread(name, 0)
			lib_pat.progress(count, total, name)
			extraction_routine("./image_features/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5), 
												[image], lbp_params[0], lbp_params[1], har_params[0], har_params[1], gab1_params, gab2_params)
			count += 1
			# filled_len = int(round(bar_len * count / float(total)))

			# percents = round(100.0 * count / float(total), 1)
			# bar = '=' * filled_len + '-' * (bar_len - filled_len)

			# sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', name))
			# sys.stdout.flush()
	print("")

def extract2 (start, end):
	path = './proyecto_features/*.npy'    
	files = glob.glob(path) 
	for name in files:
		if (name == "./proyecto_features/very_huge_raw_features.npy"):
			feats = np.load(name)

	for i in range(len(feats)):
		number = (i%240)+1
		group = (i//240)+1
		arr_name = "./image_features2/face_" + str(group).zfill(3) + "_" + str(number).zfill(5) + ".npy"
		np.save(arr_name, np.array([feats[i]]))

def get_feats(number_of_features):
	path = './image_features/*.npy'    
	files = glob.glob(path) 
	feats = []
	count = 0
	bar_len = 60
	total = number_of_features * 7
	for name in files:
		# print(name) 
		img = Image(name, npy = True)
		if (img.number <= number_of_features):
			count += 1
			lib_pat.progress(count, total, name)
			feat = (np.load("./image_features/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy"))
			# feats.append(np.load("./image_features/face_" + str(img.group).zfill(3) + "_" + str(img.number).zfill(5) + ".npy")[0])
			# print("FEAT")
			# print(feat)
			feats.append([feat])
			# if (feats == []):
			# 	feats = feat
			# else:
			# 	feats = np.concatenate((feats, feat))

			# print("FEATS")
			# print(feats)
			# if (count > 100):
			# 	break

			# input()
	feats = np.concatenate(feats, axis=1)
	feats = feats[0]
	print("")
	return (feats)

start = 1901
end = 2012
# end = 240
extract(start, end)
# feats = get_feats(end)
# classify(feats, end, 10, 0.85)

# start = 1
# end = 5
# extract2(start, end)
# feats = get_feats(end)
# classify(feats, end)







# path = './proyecto_features/*.npy'    
# files = glob.glob(path) 
# for name in files:
# 	print(name)
# 	if (name == "./proyecto_features/very_huge_raw_features.npy"):
# 		feat = np.load(name)
# 		print(len(feat))
# 		print(len(feat[0]))
# 		# print(len(feat[1]))
# 		print("0")
# 		print(feat[0])
# 		print("1")
# 		print(feat[1])
# 		print("FEAT")
# 		print(feat)












