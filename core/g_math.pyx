#from __future__ import division

import copy
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double exp(double)
    double sqrt(double)
    double log(double)
    double M_PI

__all__ = ['r_square', 'intervals', 'remove_nans', 'fit', 'exponential', 
           'gaussian', 'gaussian_normalized', 'sine', 'cosine', 'polynomial', 
           'poisson', 'pca', 'average_path_direction', 'hist_pmf', 'logit',
           'inverse_logit']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cpdef logit(double x):
    """Implements the logit function, i.e. ln(x/ (1-x))
    INPUT: x
    OUTPUT: logit(x)
    """
    return log(x / (1-x))
    
cpdef inverse_logit(double x):
    """Implements the inverse of the logit function, i.e. exp(x) / (1 + exp(x))
    INPUT: x
    OUTPUT: logit^{-1}(x)
    """
    return exp(x) / (1 + exp(x))

def pca(data, fpcOnly=True):
    """Performs a principal component analysis of a dataset.
    INPUT: data: Numpy array or list. The individual datapoints are expected to 
                 be stacked along axis 0.
           fpcOnly: Boolean. Return only the first principal component or the 
                    full output of numpy's singular-value-decomposition 
                    algorithm?
    OUTPUT: Either the first principal component of the data (a vector of 
            magnitude 1.0) , or U, s, Vh as described in numpy.svd()
    WARNING: scipy's singular value decomposition may produce incorrect results
             in some cases (look for fixes in future versions and remove this
             warning)!
    """
    data = np.array(data)
    # Compute center of mass:
    com = np.mean(data, axis=0)
    # Singular value decomposition:
    uu, dd, vv = np.linalg.svd(data - com)
    # Now vv[0] contains the first principal component, i.e. the direction 
    # vector of the 'best fit' line in the least squares sense. Its magnitude is
    # 1.0
    if fpcOnly:
        return vv[0]
    else:
        return uu, dd, vv    


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def average_path_direction(points):
    """Computes the average direction of a set of points, which trace out a 
    path. It is a length-weighted average, i.e. the direction of the connecting
    vector between two individual points is weighted stronger, the larger its 
    norm.
    INPUT: points: Array of (2D or 3D) points that trace out a path.
    OUTPUT: normalized 
    """
    points = np.array(points)
    diffVectors = points[1:] - points[:-1]
    avgDir = np.average(diffVectors, axis=0, 
                        weights=map(np.linalg.norm, diffVectors))
    return avgDir / np.linalg.norm(avgDir)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def fit(fitfnc, p0, data):
    """Least-squares fitting of a given function to data (adjusted from Scipy 
    Cookbook).
    INPUT: function: Function to fit.
           parameters: Initial guess of the function's parameters.
           data: 1D or 2D array holding the data to fit, i.e. [x] and [x,y] 
                 respectively.
    OUTPUT: y: The solution (or the result of the last iteration for an 
               unsuccessful call).    
            success: Integer reporting on the success of the fitting procedure. 
            r_square: The coefficient of determination of the fit.                
    """

    if len(data) == 1:
        data = np.array([np.linspace(data.shape[0]), data])
    else:
        data = map(np.array, data)
        
    def errfnc(p, x, y):
        return fitfnc(p, x) - y
        
    p, success = optimize.leastsq(errfnc, p0, args=tuple(data))
    return p, success, r_square(data[1], fitfnc(p, data[0])) 


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def r_square(observed,predicted):
    """ Computes the coefficient of determination (r_square) of fit values to
    data. This is an indication of how well the fit is explaining the variation
    of the data.
    INPUT: observed: List of data values.
           predicted: List of modeled values that correspond to the data.
    OUTPUT: r_square: Coefficient of determination         
    """
    
    # Total sum of squares:
    avg = sp.mean(observed)
    sstot = sum([(y-avg)**2. for y in observed])    
    # Sum of squared errors:
    sserr = sum([(z[0]-z[1])**2. for z in zip(observed,predicted)])
    
    return 1. - sserr/sstot
    
    
#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

def exponential(d=None):
    """Returns an exponential function with default parameters.
    INPUT: d: Dictionary supplying any fixed parameters. The key indicates the 
              index of the fixed parameter, the value its numerical value. Any
              parameter not supplied via d is a free parameter in the returned
              function.
    OUTPUT: Function object and default parameters.
    """
    if d is None:
        d = {}
    p0 = np.array([1.0, 1.0, 0.0, 0.0])
    pIndices = np.array([k for k in range(len(p0)) if k not in d.keys()])
    if len(pIndices > 0):
        p0 = tuple(p0[pIndices])
    else:
        p0 = ()
    def expon(p, x):
        for i, pi in enumerate(pIndices):
            d[pi] = p[i]    
        return d[0] * np.exp(d[1] * (x+d[2])) + d[3]
               
    return expon, p0

#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

def gaussian(d=None):
    """Returns a Gaussian function with default parameters.
    INPUT: d: Dictionary supplying any fixed parameters. The key indicates the 
              index of the fixed parameter, the value its numerical value. Any
              parameter not supplied via d is a free parameter in the returned
              function.
    OUTPUT: Function object and default parameters.
    """
    if d is None:
        d = {}
    p0 = np.array([1.0, 0.0, 1.0])
    pIndices = np.array([k for k in range(len(p0)) if k not in d.keys()])
    if len(pIndices > 0):
        p0 = tuple(p0[pIndices])
    else:
        p0 = ()
    def gauss(p, x):
        for i, pi in enumerate(pIndices):
            d[pi] = p[i]
        return d[0] * np.exp(-0.5 * ((x - d[1]) / d[2])**2.0)
    return gauss, p0

#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

def gaussian_normalized(d={}):
    """Returns a normalized Gaussian function with default parameters.
    INPUT: d: Dictionary supplying any fixed parameters. The key indicates the 
              index of the fixed parameter, the value its numerical value. Any
              parameter not supplied via d is a free parameter in the returned
              function.
    OUTPUT: Function object and default parameters.
    """
    if d is None:
        d = {}
    p0 = np.array([0.0, 1.0])
    pIndices = np.array([k for k in range(len(p0)) if k not in d.keys()])
    if len(pIndices > 0):
        p0 = tuple(p0[pIndices])
    else:
        p0 = ()
    def gauss(p, x):
        for i, pi in enumerate(pIndices):
            d[pi] = p[i]    
        return np.exp(-0.5 * ((x - d[0]) / d[1])**2.0) / \
               (np.sqrt(2.0 * np.pi) * d[1])
    return gauss, p0


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

def poisson():
    """Returns a Poisson distribution function with default parameters.
    INPUT: None
    OUTPUT: Function object and default parameters.
    """
    def pois(p, x):    
        return p**x * np.exp(-p) / sp.factorial(x) 
    return pois, 1.0            


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

def sine():
    """Returns a Sine function with default parameters.
    INPUT: None
    OUTPUT: Function object and default parameters.
    """
    def sin(p, x):
        return p[0] + p[1] * np.sin(p[2] * (x + p[3]))
    return sin, (0.0, 1.0, 1.0, 0.0)


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------


def cosine():
    """Returns a Cosine function with default parameters.
    INPUT: None
    OUTPUT: Function object and default parameters.
    """
    def cos(p, x):
        return p[0] + p[1] * np.cos(p[2] * (x + p[3]))
    return cos, (0.0, 1.0, 1.0, 0.0)


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

def polynomial(order):
    """Returns a Polynomial function with default parameters.
    INPUT: order: The order of the polynomial function to be constructed.
    OUTPUT: Function object and default parameters.
    """
    def pol(p, x):
        x = np.array(x)
        sum = np.zeros_like(x)        
        for i in range(order+1):
            sum = sum + p[i] * x**i
        return sum
    return pol, [1.0 for i in xrange(order+1)]


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------


def intervals(start,stop,nIntervals):
    """Generates a list of intervals from a value range.
    INPUT: start: The lower end of the range.
           stop: The upper end of the range.
           nIntervals: The number of intervals in which the range should be
                       divided.
    OUTPUT: List of intervals.
    EXAMPLE: intervals(0,10,2) divides the range [0,10] in two intervals: (0,5)
             and (5,10). 
    """

    return zip(sp.linspace(start,stop,nIntervals+1)[:-1],
               sp.linspace(start,stop,nIntervals+1)[1:])


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------


def remove_nans(a):
    """Removes NaN  (not a number) entries from an array. If the array contains
    sub-arrays, all sub-arrays containing NaNs are removed.
    INPUT: a: array
    OUTPUT: c: array with NaNs removed
    """

    if type(a[0]) == []:
        c = filter(lambda x: not any(sp.isnan(x)), a)
    else:
        c = filter(lambda x: not sp.isnan(x), a)
    return c


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------


def hist_pmf(x, bins=10, newFigure=True, **kwargs):
    """Plots a normalized histogram of data. The normalization is performed such
    that the bin values form a probability mass function (i.e. add up to one).
    INPUT:
        x: x are the data to be binned. x can be an array, a 2D array with 
           multiple data in its columns, or a list of arrays with data of 
           different length.       
        bins: Either an integer number of bins or a sequence giving the bins.
        newFigure: Use the current figure or create a new one? (Boolean.)
        **kwargs
            The keyword arguments are passed on to the function np.bar() which 
            does the actual plotting. Refer to its documentation for details.
    OUTPUT:
        b: Array of matplotlib.patches that make up the histogram / bar plot.
    """
    h = np.histogram(x, bins, normed=False)
    h0normed = h[0]/float(sum(h[0]))
    if newFigure:
        plt.figure()
    b = plt.bar(h[1][:-1], h0normed, width=h[1][1:]-h[1][:-1])
    return (h0normed, h[1], b)


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

class Quaternion(object):
    """Simple Quaternion implementation"""

    def __init__(self, *q):
        if len(q) == 1:
            q = q[0]
        self.q = np.array(q)
        
    def _q(self):
        return self.q.copy()    

    def _scalar(self):
        return self.q[0]
        
    def _vector(self):
        return self.q[1:]            

    @staticmethod
    def from_rotation_axis(angle, axis):
        """Creates a quaternion from rotation axis and angle of rotation"""
        if angle == 0.0:
            return Quaternion(1., 0., 0., 0.)
        else:
            axis = np.array(axis) / np.linalg.norm(axis)
            halfangle = angle / 2.0
            w = np.cos(halfangle)
            xyz = axis * np.sin(halfangle)
            return Quaternion(w, *xyz)
            
    @staticmethod
    def from_two_vectors(vFrom, vTo):
        dp = np.dot(vFrom, vTo)
        axis = np.cross(vFrom, vTo)
        w = np.linalg.norm(vFrom) * np.linalg.norm(vTo) + dp
        if w < 1e-6: # vectors span 180 degrees
            return Quaternion(0., -vFrom[2], vFrom[1], vFrom[0])
        else:
            return Quaternion(w, *axis)._normalize()
        #return Quaternion.from_rotation_axis(np.arccos(dp/(np.linalg.norm(vFrom)*np.linalg.norm(vTo))))        

    def rotate(self, vector):
        if not isinstance(vector, Quaternion):
            vector = Quaternion(0., *vector)
            return (self * vector * self._reciprocal())._vector()
        else:
            return self * vector * self._reciprocal()    
        
    def __mul__(self, other):
        """Quaternion - quaternion multiplication. Note: noncommutative!"""
        w1, x1, y1, z1 = self._q()
        w2, x2, y2, z2 = other._q()    
        return Quaternion(w1*w2 - x1*x2 - y1*y2 - z1*z2,
                          w1*x2 + x1*w2 + y1*z2 - z1*y2,
                          w1*y2 - x1*z2 + y1*w2 + z1*x2,
                          w1*z2 + x1*y2 - y1*x2 + z1*w2)
                               
    def __rmul__(self, other):
        """Scalar - quaternion multiplication"""
        return Quaternion(self._q() * other)
        
    def __div__(self, other):
        """Quaternion - scalar division"""
        return Quaternion(self._q() / other)    

    def _normalize(self, array=None):
        """ 
        Normalize a quaternion or array
        """
        if array is not None:
            array = np.array(array)
            return array / np.linalg.norm(array)
        else:
            return Quaternion(self._q() / self._norm())           
       
    def _norm(self):
        """Returns the norm of the quaternion"""
        return np.linalg.norm(self._q())   
    
    def _conjugate(self):
        """Returns the conjugate of the quaternion.
        """
        return Quaternion(self._scalar(), *(-self._vector()))
        
    def _reciprocal(self):
        """Returns the reciprocal of the quaternion, i.e. q^{-1}"""
        q = self._conjugate()._q()
        return Quaternion(q / self._norm()**2.)
    
    def __eq__(self, other):
        return all(self.q == other.q)
    
    def __ne__(self, other):
        return not self.__eq__(other)        
        

