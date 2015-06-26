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

def run():

	#generate synthetic data
	x, y, w, beta = sd(n=1000, p=50, k=1, sp_beta=0.9, sp_alpha=0.9)
	tr_x, tst_x, tr_y, tst_y = cv.train_test_split(x, y, test_size=0.2)
	
	#Train the fhim model.
	fhim = FHIM(lbd_beta=100, lbd_alpha=100)
	fhim.fit(tr_x, tst_x, tr_y, tst_y, KK=5, debug=True)
	return

if __name__ == '__main__':
	
	run()
	#cProfile.run('run()')
	#para_run()
