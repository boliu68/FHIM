import datetime as dt
import numpy as np
import pickle
from scipy import sparse
from scipy.optimize import line_search

def interact_predict(x, a):
	if sparse.issparse(x):
		sum_ax = x.dot(a)
		x2 = x.copy()
		x2.data = x2.data ** 2
		sum_a2x2 = x2.dot(a ** 2)
		interact_pred = np.sum(sum_ax ** 2 - sum_a2x2, 1)
	else:
		sum_ax = x.dot(a)
		x2 = x ** 2
		sum_a2x2 = x2.dot(a ** 2)
	
		interact_pred = np.sum(sum_ax ** 2 - sum_a2x2, 1)

	return interact_pred

def ls_loss_func(var, fhim, x, y, var_name):

	#prediction
	if var_name == 'a':
		beta = fhim.beta
		xk = fhim.a
		set_zero = np.multiply(xk[:,-1], var) < 0

		a = var
		a[set_zero] = 0
		a = a.reshape((-1, 1))

	elif var_name == 'beta':
		a = fhim.a[:, -1].reshape((-1,1))
		xk = fhim.beta
		beta = var
		set_zero = np.multiply(xk, beta) < 0
		beta[set_zero] = 0
	else:
		print "Wrong Variable Name"
	
	interact_pred = interact_predict(x, a) + fhim.tr_fix_interact
	if sparse.issparse(x):
		pred = fhim.beta0 + x.dot(beta) + interact_pred
	else:
		pred = fhim.beta0 + np.dot(x, beta) + interact_pred

	return np.linalg.norm(y - pred) ** 2
