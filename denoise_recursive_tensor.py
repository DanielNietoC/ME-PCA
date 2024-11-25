import numpy as np
    
def denoise_recursive_tensor(data = None,window = None,varargin = None): 
    # MP-PCA based denoising of multidimensional (tensor structured) data.
    
    # Usage is free but please cite Olesen, JL, Ianus, A, Østergaard, L,
# Shemesh, N, Jespersen, SN. Tensor denoising of multidimensional MRI data.
# Magn Reson Med. 2022; 1- 13. doi:10.1002/mrm.29478
    
    ###########################################################################
# Input variables.
    
    # data: data with noise window: window to pass over data with. Typically
# voxels.
    
    # window = [5 5] would process patches of data with dimensions 5x5x...
    
    # varargin: is specified as name-value pairs (i.e. ...,'mask',mask,...)
#             indices: determines tensor-struture of patches. For instance
#             for data with 5 indices with the three first being voxels,
#             indices = {1:3 4 5} will combine the voxels into one index
#             and retain the others so that each patch is denoised as a
#             three-index tensor -- indices = {1:2 3 4 5} would denoise
#             each patch as a four-index tensor and so on. It defaults to
#             combining the voxel/window indices and sorting according to
#             index dimensionality in ascending order, since this appears
#             to be optimal in most cases. mask: if a logical mask is
#             specified, locations were the sliding window contains no
#             "true" voxels are skipped. opt_shrink: uses optimal shrinkage
#             if true (default is false) sigma: specifies a know value for
#             the noise sigma rather than estimating it using
#             MP-distribution
    
    # Output:
    
    # denoised: the denoised data
    
    # Sigma2: estimated noise variance
    
    # P: estimated number signal components
    
    # SNR_gain: an estimate of the estimated gain in signal-to-noise ratio.
###########################################################################
# reshape data to have all voxels and measurement indices along two indices
    dims = data.shape
    dims_vox = dims(np.arange(1,len(window)+1))
    if np.asarray(dims_vox).size == 1:
        dims_vox[2] = 1
    
    num_vox = np.prod(dims_vox)
    data = reshape(data,num_vox,[])
    # determine default index ordering (same order as input data and all of
# them with window indices combined in one)
    vox_indices = np.arange(1,len(window)+1)
    mod_indices = np.arange(len(window) + 1,len(dims)+1)
    # get optional input
    options.indices = cat(2,np.array([vox_indices]),num2cell(mod_indices))
    options.mask = True(dims_vox)
    options.center_assign = False
    options.stride = np.ones((1,len(window)))
    options.test = False
    for n in np.arange(1,len(varargin)+2,2).reshape(-1):
        setattr(options,varargin[n],varargin[n + 1])
    
    indices = options.indices
    stride = reshape(options.stride,1,[])
    assert_(np.all(options.mask.shape == dims_vox),'mask dimensions do not match data dimensions')
    # dimensions of X array
    window = reshape(window,1,[])
    array_size = np.array([window,dims(np.arange(len(window) + 1,end()+1))])
    array_size = cat(2,array_size,np.ones((1,np.amax(cell2mat(indices)) - len(dims))))
    # index addition vector (indices of voxels within sliding window reltive to
# corner index)
    window_subs = cell(1,len(window))
    window_subs[:] = ind2sub(window,np.arange(1,np.prod(window)+1))
    index_increments = - 1 + sub2ind(dims,window_subs[:])
    # permutation order in accordance with provided indices
    permute_order = cell2mat(indices)
    permute_order = cat(2,permute_order,setdiff(np.arange(1,len(array_size)+1),permute_order))
    # size after reshaping in accordance with provided indices
    new_size = np.zeros((1,np.asarray(indices).size + 1))
    for n in np.arange(1,np.asarray(indices).size+1).reshape(-1):
        new_size[n] = np.prod(array_size(indices[n]))
    
    new_size[end()] = np.prod(array_size) / np.prod(new_size(np.arange(1,end() - 1+1)))
    new_size[new_size == 1] = []
    new_size[end() + 1] = 1
    # pre-allocate
    denoised = np.zeros((data.shape,'like',data))
    count = np.zeros((num_vox,1))
    Sigma2 = np.zeros((num_vox,1))
    P = np.zeros((num_vox,1))
    # loop over window positions and denoise
    for i in np.arange(1,num_vox+1).reshape(-1):
        # check if sliding window is within bounds
        index_vector = get_index_vector(dims_vox,i)
        if np.any(index_vector - 1 + window > dims_vox):
            continue
        # simply skip to next position if this one does not correspond to
# correct stride
        if np.any(rem(index_vector - 1,stride)):
            continue
        # indices of voxels within window
        vox_indices = i + index_increments
        # skip if no voxels in mask are included
        maskX = options.mask(vox_indices)
        if nnz(maskX) == 0:
            continue
        if options.center_assign:
            center_subs = num2cell(np.ceil(window / 2))
            center_ind = sub2ind(window,center_subs[:])
            if not maskX(center_ind) :
                continue
        # Create data matrix
        X = np.reshape(data(vox_indices,:), tuple(array_size), order="F")
        X = permute(X,permute_order)
        X = np.reshape(X, tuple(new_size), order="F")
        # denoise X
        X,sigma2,p = denoise_array_recursive_tensor(X,'num_inds',np.amin(len(X.shape),np.asarray(indices).size),varargin[:])
        X = np.reshape(X, tuple(array_size(permute_order)), order="F")
        X = ipermute(np.reshape(X, tuple(array_size(permute_order)), order="F"),permute_order)
        # assign
        if options.center_assign:
            X = reshape(X,np.asarray(vox_indices).size,[])
            X = X(center_ind,:)
            vox_indices = vox_indices(center_ind)
        denoised[vox_indices,:] = denoised(vox_indices,:) + reshape(X,np.asarray(vox_indices).size,[])
        count[vox_indices] = count(vox_indices) + 1
        Sigma2[vox_indices] = Sigma2(vox_indices) + sigma2
        P[vox_indices,np.arange[1,len[p]+1]] = P(vox_indices,:) + p
    
    # assign to skipped voxels
    skipped_vox = count == 0
    denoised[skipped_vox,:] = data(skipped_vox,:)
    
    Sigma2[skipped_vox] = nan
    P[skipped_vox] = nan
    # divided by number of times each voxel has been visited to get average
    count[skipped_vox] = 1
    
    denoised = denoised / count
    Sigma2 = Sigma2 / count
    P = P / count
    # estimate SNR gain according to removed variance
    if P.shape[2-1] == 1:
        SNR_gain = np.sqrt(np.prod(new_size) / (P ** 2 + np.sum(np.multiply((new_size(np.arange(1,end() - 1+1)) - P),P), 2-1)))
    else:
        SNR_gain = np.sqrt(np.prod(new_size) / (np.prod(P, 2-1) + np.sum(np.multiply((new_size(np.arange(1,end() - 1+1)) - P),P), 2-1)))
    
    # adjust output to match input dimensions
    denoised = np.reshape(denoised, tuple(dims), order="F")
    Sigma2 = np.reshape(Sigma2, tuple(dims_vox), order="F")
    P = np.reshape(P, tuple(np.array([dims_vox,P.shape[2-1]])), order="F")
    SNR_gain = np.reshape(SNR_gain, tuple(dims_vox), order="F")
    return denoised,Sigma2,P,SNR_gain
    
    
def get_index_vector(dims_vox = None,i = None): 
    index_vector = cell(1,np.asarray(dims_vox).size)
    index_vector[:] = ind2sub(dims_vox,i)
    index_vector = cell2mat(index_vector)
    return index_vector
    
    return denoised,Sigma2,P,SNR_gain