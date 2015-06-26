import math
import pickle
import numpy as np
import datetime as dt
from scipy import sparse
from line_search import line_search_armijo
import fhim_linesearch as ls

class FHIM:

	def __init__(self, loss='linear', lbd_beta=1, lbd_alpha=1, chacheW=False, opt_param={
	'm':20,
	'max_iter':1000,
	'tol':1e-4
	}):
		self.loss = loss
		self.lbd_beta = lbd_beta
		self.lbd_alpha = lbd_alpha

		self.a = 0
		self.K = 1
		self.beta = 0
		self.beta0 = 0

		self.isfit = False
		self.tr_dim = 1

		self.opt_param = opt_param

	def predict(self, x):

		if not self.isfit:
			#the model is not yet learnt
			print "FHIM is not yet trained."
			return

		data_dim = x.shape[1]
		if self.tr_dim != data_dim:
			print "The dimension of training and test is not compatable"
			return

		#simplify and reduce the time complexity
		#refered to factorization machine
		if sparse.issparse(x):
			sum_ax = x.dot(self.a)
			x2 = x.copy()
			x2.data = x2.data ** 2
			sum_a2x2 = x2.dot(self.a ** 2)
			interact_pred = np.sum(sum_ax ** 2 - sum_a2x2, 1)
			pred = self.beta0 + x.dot(self.beta) + interact_pred
		else:
			sum_ax = x.dot(self.a)
			x2 = x ** 2
			sum_a2x2 = x2.dot(self.a ** 2)
			interact_pred = np.sum(sum_ax ** 2 - sum_a2x2, 1)
			pred = self.beta0 + np.dot(x, self.beta) + interact_pred
		if self.loss == 'logistic':
			pred = 1.0 / (1.0 + np.exp(- pred))

		return pred

	def loss_func(self, x, y):
		#Return the loss without the regularization term
		pred = self.predict(x)

		return np.linalg.norm(y - pred) ** 2

	def l_bfgs_d(self, grad, s, y):

		#the max number of stored vector
		k = len(s)

		if k == 0:
			return grad

		q = grad
		
		alphas = np.zeros(k)
		rho = [np.dot(y[i], s[i]) for i in range(k)]

		for i in range(k):
			alphas[k-i-1] = np.dot(s[k-i-1], q) / rho[k-i-1]
			q = q - alphas[k-i-1] * y[k-i-1]

		r = (np.dot(s[k-1], y[k-1]) / np.dot(y[k-1], y[k-1])) * sparse.eye(self.tr_dim)
		r = r.dot(q)

		for i in range(k):
			beta = rho[i] * np.dot(y[i], r)
			r = r + (alphas[i] - beta) * s[i]
		
		#modification to owd with projection
		set_zero = np.multiply(r, grad) <= 0
		r[set_zero] = 0

		#-r is the descent direction
		return r

	def gradient(self, x, y, variable):

		#the pusedo gradient
		if self.loss == 'linear':
			pred = self.predict(x)
			err = (y - pred).reshape((-1,1)) #n * 1

			if variable == 'beta':

				if sparse.issparse(x):
					#err_diag = sparse.lil_matrix((x.shape[0], x.shape[0]))
					#err_diag.setdiag(err.flatten())
					err_diag = sparse.diags(err.flatten(), 0, format='csr')
					gd = - err_diag * x
					gd = np.asarray(gd.sum(0)).flatten()
				else:
					gd = - x * err # n * d multiply n * 1
					gd = np.sum(gd, 0).flatten()
				lbd = self.lbd_beta
				var = self.beta

			if variable == 'a':

				if sparse.issparse(x):
					xa = x.dot(self.a[:,-1])
					#xa_diag = sparse.lil_matrix((len(xa), len(xa)))
					#xa_diag.setdiag(xa)
					xa_diag = sparse.diags(xa, 0, format='csr')
					xxa = xa_diag * x
					#err_diag = sparse.lil_matrix((x.shape[0], x.shape[0]))
					#err_diag.setdiag(err.flatten())
					err_diag = sparse.diags(err.flatten(), 0, format='csr')
					gd = - err_diag * xxa
					gd = np.asarray(gd.sum(0)).flatten()
				else:
					gd = - ((x * np.dot(x, self.a[:, -1]).reshape((-1,1))) * err)
					gd = np.sum(gd, 0)
				lbd = self.lbd_alpha
				var = self.a[:, -1]

			#gd = gd / np.linalg.norm(gd)
			id1 = var > 0
			id2 = var < 0
			id3 = np.logical_and(var == 0, gd < -lbd)
			id4 = np.logical_and(var == 0, gd > lbd)
			id5 = np.logical_and(var == 0, np.logical_and(gd >= -lbd, gd <= lbd))
			#print "Mean Gradient:%f, Norm Gradient:%f, max grad:%f" % (np.min(gd), np.linalg.norm(gd), np.max(np.abs(gd)))

			gd[id1] = gd[id1] + lbd
			gd[id2] = gd[id2] - lbd
			gd[id3] = gd[id3] + lbd
			gd[id4] = gd[id4] - lbd
			gd[id5] = 0

		return gd
	
	def update_save(self, s, e):
		#Store the vetor of gradient and loss
		#If length is larger than m, then pop the first
		m = self.opt_param['m']

		if len(s) < m:
			s.append(e)
		else:
			s.pop(0)
			s.append(e)

		return s

	def update_beta(self, tr_x, tr_y):

		s_beta = []
		y_beta = []
		not_converge = True
		old_loss = [self.loss_func(tr_x, tr_y)]
		d_beta = 0

		while not_converge:

			#print "Update Beta"
			grad = self.gradient(tr_x, tr_y, 'beta')
			d_beta = self.l_bfgs_d(grad, s_beta, y_beta)
			
			#Normalize the steepest direction
			#The very large magnitude causes the line search problematic
			d_beta = d_beta / max(np.linalg.norm(d_beta), 1)

			#line search of d_beta
			step_size_beta = line_search_armijo(ls.ls_loss_func, xk=self.beta, pk=-d_beta, gfk=grad, old_fval=old_loss[-1], args=(self, tr_x, tr_y, 'beta'), alpha0=1)
			#print "Step Size:%f, Func Called:%d" % (step_size_beta[0], step_size_beta[1])
			step_size_beta = step_size_beta[0]

			if step_size_beta == None:
				#Line search cannot find step size any more
				break
			
			beta_t = self.beta - step_size_beta * d_beta
			beta_t[np.multiply(beta_t, self.beta) < 0] = 0
			s_last = beta_t - self.beta
			self.beta = beta_t

			grad_t = self.gradient(tr_x, tr_y, 'beta')
			y_last = grad_t - grad

			if np.dot(s_last, y_last) > 0:
				s_beta = self.update_save(s_beta, s_last)
				y_beta = self.update_save(y_beta, y_last)
			
			old_loss.append(self.loss_func(tr_x, tr_y))
			
			#print "Objective Function Upd Beta:%f " % old_loss[-1]
			if len(old_loss) > 10:
				if (old_loss[-11] - old_loss[-1]) / 10.0 < self.opt_param['tol'] * old_loss[-1]: 
					not_converge = False
			#if np.dot(s_beta[-1], y_beta[-1]) == 0:
					#not_converge = False
					#print "Zero Break"
			#		pass

	def update_a(self, tr_x, tr_y):
	
		s_a = []
		y_a = []
		not_converge = True
		old_loss = [self.loss_func(tr_x, tr_y)]

		while not_converge:
			#tempory save the last a
			a_last = self.a[:,-1]
			#print "Update A"
			
			###L-BFGS to update ak###
			grad = self.gradient(tr_x, tr_y, 'a')
			d_a = self.l_bfgs_d(grad, s_a, y_a)
			d_a = d_a / max(np.linalg.norm(d_a), 1)

			step_size_a = line_search_armijo(ls.ls_loss_func, xk=a_last, pk=-d_a, gfk=grad, old_fval=old_loss[-1], args=(self, tr_x, tr_y, 'a'), alpha0 = 1)
			
			#print "Step Size:%f, Func Called:%d" % (step_size_a[0], step_size_a[1])
			step_size_a = step_size_a[0]

			if step_size_a == None:
				break

			a_t = a_last - step_size_a * d_a
			a_t[np.multiply(a_t, a_last) < 0] = 0
			s_last = a_t - a_last
			self.a[:, -1] = a_t

			grad_t = self.gradient(tr_x, tr_y, 'a')
			y_last = grad_t - grad
			if np.dot(s_last, y_last) > 0:
				s_a = self.update_save(s_a, s_last)
				y_a = self.update_save(y_a, y_last)
			old_loss.append(self.loss_func(tr_x, tr_y))

			#print "Objective Function Upd Alpha:%f , Sparisty:%f" % (old_loss[-1], np.mean(self.a == 0))
			if len(old_loss) > 10:
				if (old_loss[-11] - old_loss[-1]) / 10.0 < self.opt_param['tol'] * old_loss[-1]: #or np.dot(s_a[-1], y_a[-1]) == 0:
					not_converge = False
			#if np.dot(s_a[-1], y_a[-1]) == 0:
					#not_converge = False
					#print "Zero Break"
			#		pass

	def fit(self, tr_x, tst_x, tr_y, tst_y, KK, debug=False):

		#Training the model with training data
		#debug = True means output Training and Test RMSE in each iteraction
		
		#init the parameter
		self.isfit = True
		
		self.tr_dim = tr_x.shape[1]
		self.beta = np.zeros(self.tr_dim)
		self.beta0 = 0
		self.a = np.ones((self.tr_dim, 1))
		self.tr_fix_interact = 0

		while (self.K == 1 or (self.K > 1 and np.sum(np.abs(self.a[:, - 1])) != 0)) and (self.K <= KK):
			
			not_converge = True
			old_loss = []
			iter = 0
			print "IterK:%d, Obj Func:%f, Sparsity Beta %f, Sparsity Alp %f" % (self.K, self.loss_func(tr_x, tr_y), np.mean(self.beta == 0), np.mean(self.a == 0))

			#Repeat until convergence
			while not_converge:
			
				old_time = dt.datetime.now()
				old_fval = self.loss_func(tr_x, tr_y)

				self.update_beta(tr_x, tr_y)
				self.update_a(tr_x, tr_y)

				old_loss.append(self.loss_func(tr_x, tr_y))
				
				#####################
				####Stop Criteria####
				if len(old_loss) > 10:
					if (old_loss[-11] - old_loss[-1]) / 10.0 < self.opt_param['tol'] * old_loss[-1]:
						not_converge = False
				iter += 1

				if debug:
					tr_pred = self.predict(tr_x)
					tst_pred = self.predict(tst_x)
					def rmse(y, pred):
						return math.sqrt(np.sum((y - pred) ** 2) * 1.0 / len(y))
		
					tr_rmse = rmse(tr_y, tr_pred)
					tst_rmse = rmse(tst_y, tst_pred)

					new_time = dt.datetime.now()
					print "N:%d, P:%d, K:%d, Iter:%d, Time:%s, Obj Func:%f, Sparsity Beta %f, Sparsity Alp %f, Tr RMSE:%f, Tst RMSE:%f" % (tr_x.shape[0], tr_x.shape[1], KK, iter, str(new_time - old_time), old_loss[-1], np.mean(self.beta == 0), np.mean(self.a == 0), tr_rmse, tst_rmse)
			
			print "*" * 100
			if sparse.issparse(tr_x):
				sum_ax = tr_x.dot(self.a)
				x2 =tr_x.copy()
				x2.data = x2.data ** 2
				sum_a2x2 = x2.dot(self.a ** 2)
				self.tr_fix_interact = np.sum(sum_ax ** 2 - sum_a2x2, 1)
			else:
				sum_ax = tr_x.dot(self.a)
				x2 = tr_x ** 2
				sum_a2x2 = x2.dot(self.a ** 2)
				self.tr_fix_interact = np.sum(sum_ax ** 2 - sum_a2x2, 1)
			
			self.a = np.hstack([self.a, np.ones((self.tr_dim, 1))])
			self.K += 1
		self.a = np.delete(self.a, - 1, 1)
