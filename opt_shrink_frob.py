    
def opt_shrink_frob(vals2 = None,M = None,N = None,sigma2 = None): 
    # DOI: 10.1109/TIT.2017.2653801
# vals2 = vals2/sigma2/N; # rescale
# vals2 = ((vals2-1-M/N).^2-4*M/N)./vals2;
# vals2 = vals2*sigma2*N; # scale back
    vals2 = vals2 - 2 * (N + M) * sigma2 + (N - M) ** 2 * sigma2 ** 2.0 / vals2
    return vals2