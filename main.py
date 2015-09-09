import datetime as dt
import pickle
import sys
import numpy as np
import math
import cProfile
from FHIM import FHIM
from sklearn import cross_validation as cv
from multiprocessing import Pool
from sklearn.utils import shuffle
from scipy import sparse
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import scale
from synthetic_data import synthetic_data as sd
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import matplotlib.colors as mcolors

def run():

	#generate synthetic data
	x, y, w, beta = sd(n=1000, p=100, k=1, sp_beta=0.8, sp_alpha=0.8)
	tr_x, tst_x, tr_y, tst_y = cv.train_test_split(x, y, test_size=0.2)
	
	#Train the fhim model.
	fhim = FHIM(lbd_beta=100, lbd_alpha=100)
	fhim.fit(tr_x, tst_x, tr_y, tst_y, KK=1, debug=True)
	
	#print np.min(fhim.a)
	#fhim.a[fhim.a < 0] = 0
	ww = np.dot(fhim.a, fhim.a.T)
	w_pos = (w - np.min(w)) / (np.max(w) - np.min(w))

	#cluster w
	sc = SpectralClustering(affinity='precomputed')
	sc.fit(w_pos)
	#wc = np.array()
	#wc = np.vstack([w[sc.labels_ == i, sc.labels_ == i] for i in np.unique(sc.labels_)])

	wc = np.zeros(w.shape)
	count = 0
	for i in np.unique(sc.labels_):
		wc[count:count + np.sum(sc.labels_ == i), :] = w[sc.labels_ == i, :]
		count += np.sum(sc.labels_ == i)
	count = 0
	for i in np.unique(sc.labels_):
		wc[:, count:count + np.sum(sc.labels_ == i)] = w[:, sc.labels_ == i]
		count += np.sum(sc.labels_ == i)

	wwc = np.zeros(w.shape)
	count = 0
	for i in np.unique(sc.labels_):
		wwc[count:count + np.sum(sc.labels_ == i), :] = ww[sc.labels_ == i, :]
		count += np.sum(sc.labels_ == i)
	count = 0
	for i in np.unique(sc.labels_):
		wwc[:, count:count + np.sum(sc.labels_ == i)] = ww[:, sc.labels_ == i]
		count += np.sum(sc.labels_ == i)


	cmap = mcolors.ListedColormap([(0, 0, 1), 
                               (0, 1, 0), 
                               (1, 0, 0)])

	plt.set_cmap('bwr')
	#plt.subplot(121)
	plt.title("Groundtruth Interaction Effects", fontsize=20)
	plt.grid(True)
	plt.imshow(w, vmin=-5, vmax=5)#, cmap=cmap)
	plt.colorbar()
	plt.show()

	#plt.subplot(221)
	plt.title("Learnt Interaction Effects", fontsize=20)
	plt.imshow(ww, vmin=-5, vmax=5)#, cmap=cmap)
	plt.grid(True)
	#plt.colormap()
	plt.colorbar()
	plt.show()


	return

if __name__ == '__main__':
	
	run()
	#cProfile.run('run()')
	#para_run()
