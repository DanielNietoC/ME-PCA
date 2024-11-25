import numpy as np
    
def denoise_array_recursive_tensor(X = None,varargin = None): 
    # input:
# X: array to be denoised
    
    dims = X.shape
    # handle inputs
    options.opt_shrink = True
    options.subtract_mean = False
    options.num_inds = len(dims)
    
    options.full_sigma2_pass = True
    options.test = False
    for n in np.arange(1,len(varargin)+2,2).reshape(-1):
        setattr(options,varargin[n],varargin[n + 1])
    
    use_MPPCA = not isfield(options,'sigma2') 
    if not use_MPPCA :
        sigma2 = options.sigma2
        options.use_initial_sigma2_pass = False
    
    ## handle special case of matrix input
    if len(dims) == 2:
        if np.amin(dims) == 1:
            sigma2 = 0
            P = 1
            return X,sigma2,P
        # subtract mean from X if specified
        if options.subtract_mean:
            X,X_mean = subtract_mean(X)
        else:
            X_mean = 0
        # get singular values and vectors
        U,S,V = svd(X,'econ')
        # MP cutoff
        if use_MPPCA:
            sigma2 = estimate_noise(S,dims)
        # apply cutoff
        U,S,V,P = discard_noise_components(U,S,V,sigma2)
        # optimal shrinkage
        if options.opt_shrink:
            S = apply_optimal_shrinkage(U,S,V,sigma2)
        # reconstruct X
        X = U * S * np.transpose(V) + X_mean
        return X,sigma2,P
    
    ## estimate sigma2 from first SVD or make full HOSVD pass to get all singular values for combined sigma2 estimate
    if use_MPPCA:
        if options.full_sigma2_pass:
            num_SVDs = options.num_inds
        else:
            num_SVDs = 1
        for n in np.arange(1,num_SVDs+1).reshape(-1):
            X = reshape(X,dims(n),[])
            if options.subtract_mean:
                X,X_mean[n] = subtract_mean(X)
            else:
                X_mean[n] = 0
            U[n],S[n],V[n] = svd(X,'econ')
            __,P[n] = estimate_noise(S[n],dims)
            X = V[n] * S[n]
        sigma2,P = combined_noise_estimate(S,dims,P)
    else:
        n = 1
        X = reshape(X,dims(n),[])
        if options.subtract_mean:
            X,X_mean[n] = subtract_mean(X)
        else:
            X_mean[n] = 0
        U[n],S[n],V[n] = svd(X,'econ')
    
    ## do recursive SVD
# reuse calculations for n==1 above
    n = 1
    U[n],S[n],V[n],P[n] = discard_noise_components(U[n],S[n],V[n],sigma2)
    X = V[n] * S[n]
    
    # continue for remaining indices
    for n in np.arange(2,options.num_inds+1).reshape(-1):
        if P(n - 1) == 0:
            P = P(np.arange(1,n - 1+1))
            break
        X = reshape(X,dims(n),[])
        if options.subtract_mean:
            X,X_mean[n] = subtract_mean(X)
        else:
            X_mean[n] = 0
        U[n],S[n],V[n] = svd(X,'econ')
        U[n],S[n],V[n],P[n] = discard_noise_components(U[n],S[n],V[n],sigma2)
        if options.opt_shrink and n == options.num_inds:
            S[n] = apply_optimal_shrinkage(U[n],S[n],V[n],sigma2)
        X = V[n] * S[n]
    
    ## reconstruct denoised X
    for n in flip(np.arange(1,len(P)+1)).reshape(-1):
        if P(n) == 0:
            X = np.zeros((U[n].shape[1-1],X.shape[1-1])) + X_mean[n]
        else:
            X = U[n] * np.transpose(reshape(X,[],P(n))) + X_mean[n]
    
    X = np.reshape(X, tuple(dims), order="F")
    
    P = cat(2,P,np.zeros((1,options.num_inds - len(P))))
    
def estimate_noise(S = None,dims = None): 
    M = S.shape[1-1]
    N = np.prod(dims) / M
    
    vals2 = diag(S) ** 2
    P = np.transpose((np.arange(0,len(vals2) - 1+1)))
    
    sigma2_estimates = cumsum(vals2,'reverse') / (M - P) / (N - P)
    
    cutoff_estimates = sigma2_estimates * (np.sqrt(M) + np.sqrt(N)) ** 2
    
    P = - 1 + find(vals2 < cutoff_estimates,1)
    
    if len(P)==0:
        P = len(vals2)
        sigma2 = 0
    else:
        sigma2 = sigma2_estimates(P + 1)
    
    if P == 0 and np.amin(M,N) == 1:
        P = 1
    
    
def combined_noise_estimate(S = None,dims = None,P = None): 
    sigma2 = 0
    denominator = 0
    for n in np.arange(1,len(S)+1).reshape(-1):
        M = S[n].shape[1-1]
        N = np.prod(dims) / M
        vals2 = diag(S[n]) ** 2
        sigma2 = sigma2 + sum(vals2(np.arange(P(n) + 1,end()+1)))
        denominator = denominator + (M - P(n)) * (N - P(n))
    
    sigma2 = sigma2 / denominator
    for n in np.arange(1,len(S)+1).reshape(-1):
        M = S[n].shape[1-1]
        N = np.prod(dims) / M
        cutoff = sigma2 * (np.sqrt(M) + np.sqrt(N)) ** 2
        P[n] = nnz(diag(S[n]) ** 2 > cutoff)
    
    
def discard_noise_components(U = None,S = None,V = None,sigma2 = None): 
    M = U.shape[1-1]
    N = V.shape[1-1]
    cutoff = sigma2 * (np.sqrt(M) + np.sqrt(N)) ** 2
    P = nnz(diag(S) ** 2 > cutoff)
    U = U(:,np.arange(1,P+1))
    S = S(np.arange(1,P+1),np.arange(1,P+1))
    V = V(:,np.arange(1,P+1))
    
def apply_optimal_shrinkage(U = None,S = None,V = None,sigma2 = None): 
    if S == 0:
        return S
    
    M = U.shape[1-1]
    N = V.shape[1-1]
    P = S.shape[1-1]
    vals2 = diag(S) ** 2
    vals2 = opt_shrink_frob(vals2,np.amax(M - P,1),np.amax(N - P,1),sigma2)
    S = diag(real(np.sqrt(vals2)))
    
def subtract_mean(X = None): 
    M,N = X.shape
    if M < N:
        X_mean = mean(X,2)
    else:
        X_mean = mean(X,1)
    
    X = X - X_mean
    return X,sigma2,P