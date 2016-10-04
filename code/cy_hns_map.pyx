cimport cython
import numpy as np
from scipy.misc import logsumexp
cimport numpy as np
from libc.math cimport exp, log 
import time
from rutils import *
# from cy_exact_inference import my_logaddexp,my_1d_logsumexp
from log_sum_exp import *

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

ITYPE=np.int
ctypedef np.int_t ITYPE_t

def cy_map(	np.ndarray[DTYPE_t,ndim=2] X0,
					np.ndarray[DTYPE_t,ndim=2] X1,
					np.ndarray[DTYPE_t,ndim=2] W0,
					np.ndarray[DTYPE_t,ndim=2] W1,
					np.ndarray[DTYPE_t,ndim=1] theta_card,
					int max_cardinality):
	cdef int n = X0.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=2] phi0 = np.dot(X0,W0.T)
	cdef np.ndarray[DTYPE_t,ndim=2] phi1 = np.dot(X1,W1.T)
	cdef np.ndarray[DTYPE_t,ndim=2] cum_prod = np.zeros((n+1,n+1),dtype=DTYPE)
	cdef np.ndarray[ITYPE_t,ndim=2] alpha_trail = np.zeros((n+1,max_cardinality+1),dtype=ITYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] alpha = -1e14*np.ones((n+1,max_cardinality+1),dtype=DTYPE)
	cdef int j, b, i, k, c, d, idx, b2,idx_t
	cdef DTYPE_t v,v2
	cdef np.ndarray[ITYPE_t,ndim=2] idx_map = np.zeros((n,n+1),dtype=ITYPE)
	
	idx = 0
	for j in range(n):
		for k in range(j+1,n+1):
			idx_map[j,k] = idx
			idx += 1
	
	for j in range(n):
		cum_prod[j,j+1] = phi0[j,0]
		for k in range(j+2,n+1):
			cum_prod[j,k] = cum_prod[j,k-1] + phi0[k-1,0]
			
	for j in range(n)[::-1]:
		if j != 0:
			alpha[j,0] = phi0[j,1] + cum_prod[j+1,n] + phi1[idx_map[j,n],0]
		else:
			alpha[j,0] = cum_prod[0,n] + phi1[idx_map[0,n],0]
		alpha_trail[j,0] = n
		if j < n-1:
			if j == 0:
				b2 = j+1
			else:
				b2 = j+2
			for k in range(b2,n-1):
				b = min(max_cardinality,n-k-1)+1
				if j != 0:
					v = phi0[j,1] + cum_prod[j+1,k] + theta_card[1] + alpha[k,1] + phi1[idx_map[j,k],0]
				else:
					v = cum_prod[j,k] + theta_card[1] + alpha[k,1] + phi1[idx_map[j,k],0]
				for c in range(2,b):
					if j != 0:
						idx_t = idx_map[j,k]
						v2 = phi0[j,1] + cum_prod[j+1,k] + theta_card[c] + alpha[k,c] + phi1[idx_t,0]
						if v2 > v:
							v = v2
					else:
						idx_t = idx_map[j,k]
						v2 = cum_prod[j,k] + theta_card[c] + alpha[k,c] + phi1[idx_t,0]
						if v2 > v:
							v = v2
						
				if v > alpha[j,0]:
					alpha[j,0] = v
					alpha_trail[j,0] = k
	
		if j < n-1:
			alpha[j,1] = phi0[j,1] + phi1[idx_map[j,j+1],1] + alpha[j+1,0]
			alpha_trail[j,1] = j+1
			for k in range(j+2,n):
				v = phi0[j,1] + cum_prod[j+1,k] + phi1[idx_map[j,k],1] + alpha[k,0]
				if v > alpha[j,1]:
					alpha[j,1] = v
					alpha_trail[j,1] = k
	
		if j < n-2:
			for c in range(2,min(max_cardinality,n-j-1)+1):
				alpha[j,c] = phi0[j,1] + phi1[idx_map[j,j+1],1] + alpha[j+1,c-1]
				alpha_trail[j,c] = j+1
				for k in range(j+2,n-c+1):
					idx_t = idx_map[j,k]
					v = phi0[j,1] + cum_prod[j+1,k] + phi1[idx_t,1] + alpha[k,c-1]
					if v > alpha[j,c]:
						alpha[j,c] = v
						alpha_trail[j,c] = k
						

	b = min(max_cardinality,n-1)+1
	segs = []
	labs = []
	cur_j = 0
	prev_j = 0
	cur_c = 0
	cur_c = np.argmax(np.hstack((alpha[cur_j,0],alpha[cur_j,1:b]+theta_card[1:b])))
	m_val = np.max(np.hstack((alpha[cur_j,0],alpha[cur_j,1:b]+theta_card[1:b])))
	Y0_map = np.zeros(n)
	cs = []
	js = []
	while prev_j < n:
		cs.append(cur_c)
		js.append(cur_j)
		if cur_c == 0:
			if cur_j != 0:
				Y0_map[cur_j] = 1
				segs.append([prev_j,cur_j])
				if Y0_map[prev_j:cur_j+1].sum() < 2:
					print Y0_map
					print "here"
					keyboard()
				labs.append(1)

			prev_j = cur_j
			if cur_j != n:
				cur_j = alpha_trail[cur_j,cur_c]
				segs.append([prev_j,cur_j])
				labs.append(0)
				prev_j = cur_j
				if prev_j == n:
					continue
					
				b = min(max_cardinality,n-cur_j-1)+1
				cur_c = np.argmax(alpha[cur_j,1:b]+theta_card[1:b])+1
		else:
			Y0_map[cur_j] = 1
			cur_j = alpha_trail[cur_j,cur_c]
			cur_c = cur_c - 1
		
	for i in range(len(labs)):
		if labs[i] == 1:
			segs[i] = [segs[i][0],segs[i][1]+1]
			if segs[i][1] == n:
				segs = segs[:-1]
				labs = labs[:-1]
				break
			else:
				segs[i+1] = [segs[i+1][0]+1,segs[i+1][1]]

	return [Y0_map,[np.array(labs),np.array(segs)]],m_val
	
def cy_get_augmented_features(np.ndarray[ITYPE_t,ndim=1] Y0, np.ndarray[ITYPE_t,ndim=1] Y1):
	cdef int n = Y0.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=2] x0_aug_feats = np.zeros((n,2),dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] x1_aug_feats = np.zeros((n*(n+1)/2,2),dtype=DTYPE)
	cdef int idx, i, j, k
	
	for i in range(n):
		x0_aug_feats[i,int(Y0[i]-1)] = 1
			
	idx = 0
	for j in range(n):
		for k in range(j+1,n+1):
			if k != j+1:
				x1_aug_feats[idx,1] = x1_aug_feats[idx-1,1]
			if Y1[k-1] == 0:
				x1_aug_feats[idx,1] += 1
				
			if k == j+1:
				if j == 0:
					x1_aug_feats[idx,0] = 1.0*(Y1[j]==1)
				else:
					x1_aug_feats[idx,0] = 1.0*(Y1[j]==0)
			else:
				x1_aug_feats[idx,0] = x1_aug_feats[idx-1,0] + 1.0*(Y1[k-1]==1)
			idx+=1
			
	return x0_aug_feats,x1_aug_feats
		
	
	