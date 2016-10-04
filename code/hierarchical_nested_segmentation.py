import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad
from crf import CRF
from cy_hns_map import cy_map, cy_get_augmented_features
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import theano
import theano.tensor as tensor
# from rutils import keyboard

def symbolic_logsumexp(x, axis=None, keepdims=False):
    ''' Numerically stable theano version of the Log-Sum-Exp trick'''
    x_max = tensor.max(x, axis=axis, keepdims=True)
    preres = tensor.log(tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=keepdims))
    # return preres + x_max.reshape(preres.shape),x_max,preres
    return preres + x_max.reshape(preres.shape)
    
def segment_index_map(j,k,n):
    return j*n - j*(j+1)/2 + k - 1
    
class HierarchicalNestedSegmentation(CRF):
    
    def __init__(self,n_x0_features=1,n_x1_features=1,max_cardinality=250,alpha=0.5,backend="autograd",**kwargs):
        self.__dict__.update(locals())
        CRF.__init__(self,**kwargs)
        self.n_parameters = 2*self.n_x0_features + 2*self.n_x1_features + self.max_cardinality + 1
        self.compile()
        
    def compile(self):
        if self.backend == "autograd":
            HierarchicalNestedSegmentation.ll_g = grad(HierarchicalNestedSegmentation.autograd_log_partition_function,argnum=2)
        elif self.backend == "theano":
            # pass
            self.logZ_fun,self.g_fun = self.compile_log_partition_function_and_gradient()
        
    def vectorize_label(self,y):
        return np.hstack((y[0],self.project_segmentation(y)))
        
    def devectorize_label(self,y):
        n = y.shape[0]/2
        y0 = y[:n]
        labs = []
        segs = []
        cur_idx = 0
        cur_lab = y[n]
        for i,v in enumerate(y[n+1:]):
            if v != cur_lab:
                labs.append(cur_lab)
                segs.append((cur_idx,i+1))
                cur_lab = v
                cur_idx = i+1
        labs.append(cur_lab)
        segs.append((cur_idx,n))
        
        return [y0,[labs,segs]]
        
    def sufficient_statistics(self, x, y):
        x0,x1 = x
        y0,y1 = y
        
        n = x0.shape[0]
        
        x0_sum = np.zeros((2,x[0].shape[1]))
        for i,yi in enumerate(y0):
            x0_sum[yi] += x0[i]

        x1_sum = np.zeros((2,x[1].shape[1]))
        for lab,seg in zip(*y1):
            if lab == 1:
                last = seg[0]
                i = seg[0]
                for v in y0[seg[0]:seg[1]]:
                    if v == 1:
                        if i != seg[0]:
                            if i == n-1:
                                x1_sum[lab] += x1[segment_index_map(last,i,n)]
                                x1_sum[0] += x1[segment_index_map(n-1,n,n)]
                                last = i
                            else:
                                x1_sum[lab] += x1[segment_index_map(last,i,n)]
                                last = i
                            
                    i += 1
            else:
                if seg[0] == 0:
                    x1_sum[lab] += x1[segment_index_map(seg[0],seg[1],n)]
                else:
                    x1_sum[lab] += x1[segment_index_map(seg[0]-1,seg[1],n)]

        C = np.zeros(self.max_cardinality + 1)
        for lab,seg in zip(*y1):
            if lab == 1:
                y_sum = np.sum(y0[seg[0]:seg[1]])
                C[int(y_sum)-1] += 1
                
        return np.hstack((x0_sum.flatten(),x1_sum.flatten(),C.flatten()))
        
    def expected_sufficient_statistics(self, x, return_logZ=False):
        w = self.get_weight_vector()
        if self.backend == "autograd":
            gradient = self.ll_g(x,w)
        elif self.backend == "theano":
            gradient = self.g_fun(x[0],x[1],self.W0,self.W1,self.theta_card,self.max_cardinality)
        if return_logZ:
            return gradient,self.log_partition_function(x,w)
        else:
            return gradient
        
    def get_weight_vector(self):
        return np.hstack((self.W0.flatten(),self.W1.flatten(),self.theta_card.flatten()))
        
    def set_weights(self, w):
        b = 0
        e = b+2*self.n_x0_features
        self.W0 = w[b:e].reshape((2,self.n_x0_features))
        b = e
        e = b+2*self.n_x1_features
        self.W1 = w[b:e].reshape((2,self.n_x1_features))
        b = e
        e = b+self.max_cardinality+1
        self.theta_card = w[b:e].reshape(self.max_cardinality+1)
                
    def map_inference(self, X, return_score=False):
        if return_score:
            return [cy_map(x[0],x[1],self.W0,self.W1,self.theta_card,self.max_cardinality) for x in X]
        else:
            return [cy_map(x[0],x[1],self.W0,self.W1,self.theta_card,self.max_cardinality)[0] for x in X]
        
    def set_loss_augmented_weights(self, x, y, w):        
        # Get n and separate the labels
        n = x[0].shape[0]
        y0 = y[0]
        y1 = y[0]
        
        # Get the current parameters
        self.set_weights(w)
        
        # Augment the bottom level weights such that
        # when they are multiplied against the augmented
        # features, it will result in <w,\psi(x,y)> + \ell(y,\hat{y})
        self.W0 = np.hstack((self.W0,np.array([[2.0*self.alpha,0],[0,2.0*self.alpha]])))
        
        # Augment the top level weights
        self.W1 = np.hstack((self.W1,np.array([[2.0*(1-self.alpha),0],[0,2.0*(1-self.alpha)]])))
                
        # Calulate the augmented features
        x0_aug_feats_t,x1_aug_feats_t = cy_get_augmented_features(y0,self.project_segmentation(y))
        x0 = np.hstack((x[0],x0_aug_feats_t))       
        x1 = np.hstack((x[1],x1_aug_feats_t))
        
        return x0,x1
        
    def project_segmentation(self,y):
        n = y[0].shape[0]
        y_proj = np.zeros(n,dtype=int)
        for lab,seg in zip(*y[1]):
            y_proj[seg[0]:seg[1]] = lab
                
        return y_proj
        
    def loss(self,y,y_hat,vectorized_labels=True):
        if vectorized_labels:
            y = self.devectorize_label(y)
            y_hat = self.devectorize_label(y_hat)
        n = y[0].shape[0]
        loss = 2.0*(self.alpha) * np.sum(y[0] != y_hat[0])
        y_proj = self.project_segmentation(y)
        y_hat_proj = self.project_segmentation(y_hat)
        loss += 2.0*(1-self.alpha) * np.sum(y_proj != y_hat_proj)
        
        return loss
    
    def log_partition_function(self,x,w):
        if self.backend == "autograd":
            return self.autograd_log_partition_function(x,w)
        if self.backend == "theano":
            return self.theano_log_partition_function(x,w)
    
    def symbolic_log_partition_function(self,x0,x1,w0,w1,theta_card,max_cardinality):
        n = x0.shape[0]
        phi0 = tensor.dot(x0,w0.T)
        phi1 = tensor.dot(x1,w1.T)

        def alpha0_iteration(j,A):
            padding = tensor.zeros((j+1,))
            values = tensor.extra_ops.cumsum(A[j:,0])
            return tensor.concatenate((padding,values),axis=0)

        alpha0,_ = theano.scan(fn=alpha0_iteration,
                                 sequences=[tensor.arange(n+1)],
                                 non_sequences=[phi0])


        alpha1_0 = tensor.concatenate((tensor.zeros((1,)),-1e10*tensor.ones((max_cardinality,))),axis=0)
        padding = -1e10*tensor.ones((max_cardinality-1,))
        alpha1_1 = tensor.concatenate((tensor.ones((1,))*(phi0[0,0] + phi1[segment_index_map(0,1,n),0]), tensor.ones((1,))*(phi0[0,1] + phi1[segment_index_map(0,1,n),1]), padding),axis=0)
        alpha1 = tensor.zeros((n,max_cardinality+1))
        alpha1 = tensor.set_subtensor(alpha1[0],alpha1_0)
        alpha1 = tensor.set_subtensor(alpha1[1],alpha1_1)
        # alpha1 = tensor.stack((alpha1_0,alpha1_1))


        def symbolic_log_partition_iteration(k,alpha1,phi0,phi1,theta_card,n,max_cardinality,alpha0):
            alpha1_k0 = symbolic_logsumexp(theta_card[1:].reshape((1,max_cardinality)) + alpha1[:k-1,1:],axis=1)
            alpha1_k0 = symbolic_logsumexp(phi0[:k-1,1] + alpha0[1:k,k] + phi1[segment_index_map(tensor.arange(k-1),k,n),0] + alpha1_k0)
            tmp = tensor.ones((1,))*(alpha0[0,k] + phi1[segment_index_map(0,k,n),0])
            tmp2 = tensor.ones((1,))*alpha1_k0
            alpha1_k0 = symbolic_logsumexp(tensor.concatenate((tmp,tmp2),axis=0))

            tmp3 = tensor.min([max_cardinality,k])
            alpha1_k1p = symbolic_logsumexp((phi0[:k,1] + alpha0[1:k+1,k] + phi1[segment_index_map(tensor.arange(k),k,n),1]).reshape((k,1)) + alpha1[:k,:tmp3],axis=0)
            pad_size = max_cardinality - tmp3
            padding = -1e10*tensor.ones((pad_size,))
            alpha1_k = tensor.concatenate((alpha1_k0*tensor.ones((1,)),alpha1_k1p,padding),axis=0)

            return tensor.set_subtensor(alpha1[k],alpha1_k)
            # tmp4 = tensor.shape_padleft(alpha1_k)
            # return tensor.concatenate((alpha1,tmp4),axis=0)
            
            

        # alpha1_k0 = logsumexp(self.theta_card[1:].reshape((1,self.max_cardinality)) + alpha1[:k-1,1:],axis=1)
        # alpha1_k0 = logsumexp(phi0[:k-1,1] + alpha0[1:k,k] + phi1[segment_index_map(np.arange(k-1),k,n),0] + alpha1_k0)
        # alpha1_k0 = logsumexp(np.hstack([alpha0[0,k] + phi1[segment_index_map(0,k,n),0],alpha1_k0]))
        #
        # alpha1_k1p = logsumexp((phi0[:k,1] + alpha0[1:k+1,k] + phi1[segment_index_map(np.arange(k),k,n),1]).reshape((k,1)) + alpha1[:k,:min(self.max_cardinality,k)],axis=0)
        # alpha1_k = np.hstack((alpha1_k0,alpha1_k1p,-1e10*np.ones(self.max_cardinality - min(self.max_cardinality,k))))

        results,_ = theano.scan(fn = symbolic_log_partition_iteration,
                                outputs_info = [alpha1],
                                sequences=[tensor.arange(2,n)],
                                non_sequences=[phi0,phi1,theta_card,n,max_cardinality,alpha0])
                                
        # for k in range(2,5):
        #     alpha1 = symbolic_log_partition_iteration(k,alpha1,phi0,phi1,theta_card,n,max_cardinality,alpha0)

        alpha1 = results[-1]

        logZ = symbolic_logsumexp(theta_card[1:].reshape((1,max_cardinality)) + alpha1[1:n,1:],axis=1)
        tmp = phi0[1:n,1] + alpha0[2:n+1,n]
        tmp1 = tmp + phi1[segment_index_map(tensor.arange(1,n),n,n),0]
        tmp2 = tmp1 + logZ
        logZ = symbolic_logsumexp(tmp2)
        logZ = symbolic_logsumexp(tensor.concatenate([tensor.ones((1,))*(alpha0[0,n] + phi1[segment_index_map(0,n,n),0]),tensor.ones((1,))*logZ],axis=0))
        
        # return phi0,phi1,alpha0,alpha1,results
        return logZ
        
    def compile_log_partition_function_and_gradient(self):
        x0_s = tensor.matrix()
        x1_s = tensor.matrix()
        w0_s = tensor.matrix()
        w1_s = tensor.matrix()
        theta_card_s = tensor.vector()
        max_cardinality_s = tensor.iscalar()
        
        logZ_s = self.symbolic_log_partition_function(x0_s,x1_s,w0_s,w1_s,theta_card_s,max_cardinality_s)
        g_params = theano.grad(logZ_s,[w0_s,w1_s,theta_card_s])
        g_w = tensor.concatenate([tensor.flatten(g) for g in g_params])
        
        logZ_fun = theano.function(inputs=[x0_s,x1_s,w0_s,w1_s,theta_card_s,max_cardinality_s],outputs=logZ_s)
        g_fun = theano.function(inputs=[x0_s,x1_s,w0_s,w1_s,theta_card_s,max_cardinality_s],outputs=g_w)
        
        return logZ_fun,g_fun
        
    def theano_log_partition_function(self,x,w):
        self.set_weights(w)
        return self.logZ_fun(x[0],x[1],self.W0,self.W1,self.theta_card,self.max_cardinality)
        
    def autograd_log_partition_function(self,x,w):
        self.set_weights(w)
        n = x[0].shape[0]
        alpha0 = []
        alpha1 = np.zeros((n+1,self.max_cardinality+1))
        tmp_arr = np.zeros((n+1)*self.max_cardinality)
        # tmp_arr = np.zeros((n+1)*self.max_cardinality)
        phi0 = np.dot(x[0],self.W0.T)
        phi1 = np.dot(x[1],self.W1.T)

        for j in range(n):
            alpha0.append(np.hstack((np.zeros(j+1),np.cumsum(phi0[j:,0]))))
        alpha0.append(np.zeros(n+1))    
        
        alpha0 = np.array(alpha0)
        
        alpha1_0 = np.hstack((np.zeros(1),-1e10*np.ones(self.max_cardinality)))
        alpha1_1 = np.hstack((phi0[0,0] + phi1[segment_index_map(0,1,n),0], phi0[0,1] + phi1[segment_index_map(0,1,n),1], -1e10*np.ones(self.max_cardinality-1)))
        alpha1 = np.vstack((alpha1_0,alpha1_1))
        for k in range(2,n):            
            alpha1_k0 = logsumexp(self.theta_card[1:].reshape((1,self.max_cardinality)) + alpha1[:k-1,1:],axis=1)
            alpha1_k0 = logsumexp(phi0[:k-1,1] + alpha0[1:k,k] + phi1[segment_index_map(np.arange(k-1),k,n),0] + alpha1_k0)
            alpha1_k0 = logsumexp(np.hstack([alpha0[0,k] + phi1[segment_index_map(0,k,n),0],alpha1_k0]))

            alpha1_k1p = logsumexp((phi0[:k,1] + alpha0[1:k+1,k] + phi1[segment_index_map(np.arange(k),k,n),1]).reshape((k,1)) + alpha1[:k,:min(self.max_cardinality,k)],axis=0)
            alpha1_k = np.hstack((alpha1_k0,alpha1_k1p,-1e10*np.ones(self.max_cardinality - min(self.max_cardinality,k))))
            alpha1 = np.vstack((alpha1,alpha1_k))

        logZ = logsumexp(self.theta_card[1:].reshape((1,self.max_cardinality)) + alpha1[1:n,1:],axis=1)
        logZ = logsumexp(phi0[1:n,1] + alpha0[2:n+1,n] + phi1[segment_index_map(np.arange(1,n),n,n),0] + logZ)
        logZ = logsumexp(np.hstack([alpha0[0,n] + phi1[segment_index_map(0,n,n),0],logZ]))
        return logZ
                
    def unnormalized_joint(self,x,y,w):
        if isinstance(y,np.ndarray):
            y = self.devectorize_label(y)
            
        return np.dot(self.sufficient_statistics(x,y),w)
    
    def deaugment(self,w):
        raise NotImplementedError("deaugment not implemented.")
        
        
##############################################################################
# Tests
##############################################################################

def test_map_inference_toy():
    x0 = np.array([-1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0]).reshape((7,1))
    x1 = []
    for j in range(7):
      for k in range(j+1,8):
          f = np.zeros(8)
          # if k-j > 1:
          f[k-j] = 1.0
          x1.append(f)
    x1 = np.array(x1)
    
    mdl = HierarchicalNestedSegmentation(max_cardinality=25)

    # Answer == 7
    mdl.W0 = np.array([-1,1],dtype=float).reshape((2,1))
    mdl.theta_card = np.zeros(26)
    mdl.W1 = np.zeros((2,8))
    assert cy_map(x0,x1,mdl.W0,mdl.W1,mdl.theta_card,mdl.max_cardinality)[1] == 7
    
    # Answer == 12
    mdl.W0 = np.array([-1,1],dtype=float).reshape((2,1))
    mdl.theta_card = np.zeros(26)
    mdl.W1 = np.zeros((2,8))
    mdl.W1[1,2] = 3.0
    assert cy_map(x0,x1,mdl.W0,mdl.W1,mdl.theta_card,mdl.max_cardinality)[1] == 12
    
    # Answer = 9
    mdl.W0 = np.array([-1,1],dtype=float).reshape((2,1))
    mdl.theta_card = np.zeros(26)
    mdl.W1 = np.zeros((2,8))
    mdl.W1[1,6] = 10.0
    assert cy_map(x0,x1,mdl.W0,mdl.W1,mdl.theta_card,mdl.max_cardinality)[1] == 9

    # Answer = 12
    mdl.W0 = np.array([-1,1],dtype=float).reshape((2,1))
    mdl.theta_card = np.zeros(26)
    mdl.W1 = np.zeros((2,8))
    mdl.W1[1,2] = 3.0
    mdl.theta_card[2] = 1.0
    assert cy_map(x0,x1,mdl.W0,mdl.W1,mdl.theta_card,mdl.max_cardinality)[1] == 12

    # Answer = 17
    mdl.W0 = np.array([-1,1],dtype=float).reshape((2,1))
    mdl.theta_card = np.zeros(26)
    mdl.W1 = np.zeros((2,8))
    mdl.W1[1,2] = 3.0
    mdl.theta_card[2] = 1.0
    mdl.theta_card[6] = 20.0
    assert cy_map(x0,x1,mdl.W0,mdl.W1,mdl.theta_card,mdl.max_cardinality)[1] == 17
    
def test_log_partition(seed,backend):
    import itertools as it
    np.random.seed(seed)
    n = 5
    n_x0_features = 2
    n_x1_features = 3
    max_cardinality = n
    x0 = np.random.randn(n,n_x0_features)
    x1 = np.random.randn(n*(n+1)/2,n_x1_features)
    x = [x0,x1]
    
    mdl = HierarchicalNestedSegmentation(max_cardinality=n,n_x0_features=n_x0_features,n_x1_features=n_x1_features,backend=backend)
    w = np.random.randn(mdl.n_parameters)
    # w = np.zeros(mdl.n_parameters)
    bf_logZ = -1e10
    for y1_l in it.product(range(2),repeat=n):
        for y0_l in it.product(range(2),repeat=n):
            y = mdl.devectorize_label(np.hstack((np.array(y0_l),np.array(y1_l))))
            valid = True
            for lab,seg in zip(*y[1]):
                if lab == 0 and not np.all(y[0][seg[0]:seg[1]] == 0):
                    valid = False
                if lab == 1:
                    if not y[0][seg[0]] == 1 or not y[0][seg[1]-1] == 1 or seg[1]-seg[0] < 2:
                        valid = False
                    
            if not valid:
                continue
            else:
                # print y1_l,y[1]
                # print y0_l
                # print "*****************************"
                bf_logZ = logsumexp([bf_logZ,mdl.unnormalized_joint(x,y,w)])
                
    mdl.set_weights(w)
    mdl_logZ = mdl.log_partition_function(x,w)
    if not np.isclose(bf_logZ,mdl_logZ):
        print bf_logZ,mdl_logZ
    assert np.isclose(bf_logZ,mdl_logZ)
    mdl.set_weights(w)
    mdl.expected_sufficient_statistics(x)
    
def gen_test_data(n_data_cases,min_len,max_len,n_features):
    np.random.seed(97330)
    X = []
    Y = []
    for i in range(n_data_cases):
        L = np.random.randint(min_len,max_len+1)
        x0 = np.random.randn(L,n_features)
        x1 = np.random.randn(L*(L+1)/2,n_features)
        X.append([x0,x1])
        
        y0 = np.random.randint(0,2,L)
        labs = []
        segs = []
        if np.all(y0 == 0):
            y1 = [np.array([0]),np.array([[0,L+1]])]
        else:
            first_pos_idx = np.min(np.where(y0==1))
            last_pos_idx = np.max(np.where(y0==1))
            if first_pos_idx != 0:
                labs.append(0)
                segs.append([0,first_pos_idx])
            labs.append(1)
            segs.append([first_pos_idx,last_pos_idx+1])
            if last_pos_idx != L-1:
                labs.append(0)
                segs.append([last_pos_idx+1,L])
            y1 = [np.array(labs),np.array(segs)]
        Y.append([y0,y1])
        
    return X,Y
    
def test_fit_ssvm():
    n_data_cases = 25
    min_len = 25
    max_len = 250
    n_features = 30
    X,Y = gen_test_data(n_data_cases,min_len,max_len,n_features)
    mdl = HierarchicalNestedSegmentation(n_x0_features=n_features,n_x1_features=n_features,max_cardinality=250,verbose=0,objective="ssvm",lambda_0=1.0)
    mdl.fit(X,Y)
    
def log_partition_time_test(case_len,grad=True,backend="autograd"):
    import time
    n_data_cases = 1
    min_len = max_len = case_len
    n_features = 30
    X,Y = gen_test_data(n_data_cases,min_len,max_len,n_features)
    mdl = HierarchicalNestedSegmentation(n_x0_features=n_features,n_x1_features=n_features,max_cardinality=250,verbose=2,objective="mle",lambda_0=0.0,backend=backend)
    w = np.random.rand(mdl.n_parameters)
    mdl.set_weights(w)
    beg = time.time()
    for i in range(1):
        if grad:
            mdl.expected_sufficient_statistics(X[0])
        else:
            mdl.log_partition_function(X[0],w)
    print case_len,':',(time.time() - beg)/1.0
    
    return (time.time() - beg)/1.0
    
def test_mle(backend):
    n_data_cases = 25
    min_len = 25
    max_len = 250
    n_features = 30
    X,Y = gen_test_data(n_data_cases,min_len,max_len,n_features)
    mdl = HierarchicalNestedSegmentation(n_x0_features=n_features,n_x1_features=n_features,max_cardinality=250,verbose=2,objective="mle",lambda_0=0.0,backend=backend)
    mdl.fit(X,Y)
    
def time_tests(backend):
    times = []
    case_lens = np.array([5,10,15,20,25,30,35,40,45,50])
    for case_len in case_lens:
        times.append(log_partition_time_test(case_len,True,backend))

    coefs = np.polyfit(case_lens,times,deg=2)
    # print coefs
    print backend,':',coefs[0] * 250**2 + coefs[1]*250 + coefs[2]
    approx_times = coefs[0] * case_lens**2 + coefs[1]*case_lens + coefs[2]
    plt.plot(case_lens,times)
    plt.plot(case_lens,approx_times)
    plt.savefig("../output/figures/%s_log_partition_runtimes.pdf"%backend)
    plt.clf()
    plt.plot(case_lens,np.log(times))
    plt.savefig("../output/figures/%s_log_partition_log_runtimes.pdf"%backend)
        
def debug_fun():
    np.random.seed(1)
    n = 5
    n_x0_features = 2
    n_x1_features = 3
    max_cardinality = n
    x0 = np.random.randn(n,n_x0_features)
    x1 = np.random.randn(n*(n+1)/2,n_x1_features)
    x = [x0,x1]

    mdl = HierarchicalNestedSegmentation(max_cardinality=n,n_x0_features=n_x0_features,n_x1_features=n_x1_features,backend="theano")
    w = np.random.randn(mdl.n_parameters)
    mdl.set_weights(w)

    x0_s = tensor.matrix()
    x1_s = tensor.matrix()
    w0_s = tensor.matrix()
    w1_s = tensor.matrix()
    theta_card_s = tensor.vector()
    max_cardinality_s = tensor.iscalar()
    
    logZ_s = mdl.symbolic_log_partition_function(x0_s,x1_s,w0_s,w1_s,theta_card_s,max_cardinality_s)
    logZ_fun = theano.function(inputs=[x0_s,x1_s,w0_s,w1_s,theta_card_s,max_cardinality_s],outputs=logZ_s,on_unused_input='ignore')
    
    res = logZ_fun(x0,x1,mdl.W0,mdl.W1,mdl.theta_card,mdl.max_cardinality)
    
    # keyboard()
    
        
if __name__=="__main__":
    # debug_fun()
    test_map_inference_toy()
    for backend in ["autograd","theano"]:
        for seed in range(5):
            test_log_partition(seed,backend)
        # time_tests(backend)
    # test_fit_ssvm()
    test_mle("theano")
    print "* Tests Passed *"