"""
    PyENT: Python version of FourmiLab's ENT: Benchmarking suite for pseudorandom number sequence.

    (c) 2020 by Minh-Hai Nguyen

"""

import numpy as np
from scipy.stats import chi2 as spChi2

def Chi2(data,bins,min_value=None,max_value=None):
    """ Compute chi-square

    bins: int or sequence of bins
    If min_value or max_value is None, use the bounds of the data
    """
    if min_value is None:
        min_value = min(data)
    if max_value is None:
        max_value = max(data)
    Os, bs = np.histogram(data,bins=bins,range=(min_value,max_value),density=False)
    E = len(data)/len(bs)
    return np.sum((Os-E)**2)/E


def Chi2Q(data,bins,min_value=None,max_value=None):
    """ Compute accumunative distribution of chi-square

    bins: int or sequence of bins
    If min_value or max_value is None, use the bounds of the data
    """
    if min_value is None:
        min_value = min(data)
    if max_value is None:
        max_value = max(data)
    Os, bs = np.histogram(data,bins=bins,range=(min_value,max_value),density=False)
    E = len(data)/(len(bs)-1)
    c2 = np.sum((Os-E)**2)/E
    return 1-spChi2.cdf(c2,len(bs)-2)


def Pi(data,min_value=None,max_value=None):
    """ Estimate the value of pi from the data by Monte-Carlo method
    by converting data into a series of (x,y) coordinates in a square
    and count the number of points fall within a circle bounded by that square

    If min_value or max_value is None, use the upper and lower bounds of the data
    """
    if min_value is None:
        min_value = min(data)
    if max_value is None:
        max_value = max(data)
    R = (max_value - min_value)/2
    Rloc = (max_value + min_value)/2
    R2 = R**2
    xs = data[:-1] - Rloc
    ys = data[1:] - Rloc
    ds = xs**2 + ys**2
    hits = np.sum(ds<R2)
    return 4*hits/len(xs)


def Entropy(data,bins,min_value=None,max_value=None):
    """ Compute Shannon Entropy of the data

    bins: int or sequence of bins
    If min_value or max_value is None, use the bounds of the data
    """
    if min_value is None:
        min_value = min(data)
    if max_value is None:
        max_value = max(data)
    Os,_ = np.histogram(data,bins=bins,range=(min_value,max_value),density=False)
    Ps = Os/len(data)
    Ps = Ps[Ps>0]
    E = -np.sum(Ps*np.log2(Ps))
    return E


def Corr(data):
    """ Serial Correlation Coefficient
    """
    result = np.corrcoef(data[:-1],data[1:])
    return abs(result[0,1]/result[0,0])


def ENT(data,bins,min_value=None,max_value=None,display=False):
    """ Comprehensive randomness Benchmarking

    bins: int or sequence of bins
    If min_value or max_value is None, use the bounds of the data
    display: print out results or not
    """
    if min_value is None:
        min_value = min(data)
    if max_value is None:
        max_value = max(data)
    # Number of bins
    if isinstance(bins,list):
        num_bins = len(bins)
    else:
        num_bins = bins
    # Shannon Entropy
    entropy = Entropy(data,bins,min_value,max_value)
    max_ent = np.log2(num_bins)
    # Chi-square Test
    chi2 = Chi2(data,bins,min_value,max_value)
    chi2Q = 1-spChi2.cdf(chi2,num_bins-1)
    # Mean
    mean = np.mean(data)
    median = (min_value + max_value)/2
    # Monte-Carlo value for Pi
    pi = Pi(data,min_value,max_value)
    # Serial Correlation
    corr = Corr(data)
    # Print out
    if display:
        print("Entropy = ", entropy, "bits per character.")
        print("Optimum compression would reduce the size of this data by %.0f percent."
                    %((max_ent-entropy)*100/max_ent))
        print()
        print("Chi square distribution is %.2e, and randomly would exceed this value %.1f percent of the times."
                    %(chi2,chi2Q*100))
        print()
        print("Arithmetic mean value is %.2e" %mean)
        print("Median value is %.2e" %median)
        print()
        print("Monte-Carlo value for Pi is ", pi, " (error %.3f percent)" %(abs(np.pi-pi)*100/np.pi))
        print()
        print("Serial correlation coefficient is ", corr)
    return entropy, chi2, pi, corr
