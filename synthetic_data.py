import numpy as np
import pickle

def synthetic_data(n=1000, p=50, k=10, sp_beta=0.95, sp_alpha=0.95):

	beta = np.random.normal(scale=1, size=p)
	ak = np.random.normal(scale=1, size=(p, k))
	id = np.arange(p)

	#sparsity
	np.random.shuffle(id)
	beta[id[0:int(sp_beta * p)]] = 0

	for kk in range(k):
		np.random.shuffle(id)
		ak[id[0:int(sp_alpha * p)], kk] = 0

	epsilon = np.random.normal(scale=0.1, size=n)

	x = np.random.normal(scale=1, size=(n, p))
	
	sum_ax = x.dot(ak)
	x2 = x.copy()
	x2 = x2 ** 2
	sum_a2x2 = x2.dot(ak ** 2)
	interact_pred = np.sum(sum_ax ** 2 - sum_a2x2, 1)
	pred = np.dot(x, beta) + interact_pred
	#pred = interact_pred

	w = np.dot(ak, ak.T)
	y = pred + epsilon
	pickle.dump(ak, open("ak.pkl", "w"))

	#print '\n'.join(["Y:%f, Pred:%f"%(y[i], epsilon[i]) for i in range(20)])

	return x, y, w, beta
