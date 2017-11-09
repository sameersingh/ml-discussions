import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat


################################################################################
## LINEAR CLASSIFY #############################################################
################################################################################


class linearClassify(classifier):
    """A simple linear classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier 
                  (1xN or CxN numpy array, where N=# features, C=# classes)

    Note: currently specialized to logistic loss
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for linearClassify object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
                      shape (1,N) for binary classification or (C,N) for C classes
        """
        self.classes = []
        self.theta = np.array([])

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"

    #@property
    #def theta(self):
    #    """Get the linear coefficients"""
    #    return self._theta
    #def theta.setter(self,theta)
    #    self._theta = np.atleast_2d(


    def __repr__(self):
        str_rep = 'linearClassify model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


    def __str__(self):
        str_rep = 'linearClassify model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


## CORE METHODS ################################################################

    ### TODO: plot2D member function?  And/or "update line" & redraw? 
    ###   ALT: write f'n to take linear classifier & plot it
    ###   pass into gradient descent f'n as "updateFn" ?

    def predictSoft(self, X):
        """
        This method makes a "soft" linear classification predition on the data
        Uses a (multi)-logistic function to convert linear response to [0,1] confidence

        Parameters
        ----------
        X : M x N numpy array 
            M = number of testing instances; N = number of features.  
        """
        theta,X = twod(self.theta), arr(X)          # convert to numpy if needed
        resp = theta[:,0].T + X.dot(theta[:,1:].T)  # linear response (MxC)
        prob = np.exp(resp)
        if resp.shape[1] == 1:       # binary classification (C=1)
            prob /= prob + 1.0       # logistic transform (binary classification; C=1)
            prob = np.hstack( (1-prob,prob) )  # make a column for each class
        else:
            prob /= np.sum(prob,axis=1)   # normalize each row (for multi-class)

        return prob

    """
    Define "predict" here if desired (or just use predictSoft + argmax by default)
    """

    def train(self, X, Y, reg=0.0, initStep=1.0, stopTol=1e-4, stopIter=5000, plot=None):
        """
        Train the linear classifier.  
        """
        self.theta,X,Y = twod(self.theta), arr(X), arr(Y)   # convert to numpy arrays
        M,N = X.shape
        X1 = np.hstack((np.ones((M,1)),X))     # make data array with constant feature
        if Y.shape[0] != M:
            raise ValueError("Y must have the same number of data (rows) as X")
        self.classes = np.unique(Y)
        if len(self.classes) != 2:
            raise ValueError("Y should have exactly two classes (binary problem expected)")
        if self.theta.shape[1] != N+1:         # if self.theta is empty, initialize it!
            self.theta = np.random.randn(1,N+1)
        Y01 = toIndex(Y, self.classes)         # convert Y to "index" (binary: 0 vs 1)

        it   = 0
        done = False
        Jsur = []
        J01  = []
        while not done:
            step = (2.0 * initStep) / (2.0 + it)   # common 1/iter step size change

            for i in range(M):  # for each data point
                # compute linear response:
                respi = self.theta[:,0] + twod(X[i,:]).dot(self.theta[:,1:].T) 
                yhati = 1.0 if respi > 0 else 0.0   # convert to 0/1 prediction
                sigx  = np.exp(respi) / (1.0+np.exp(respi))
                gradi = -Y01[i]*(1-sigx)*twod(X1[i,:]) + (1-Y01[i])*sigx*twod(X1[i,:]) + reg*self.theta
                self.theta = self.theta - step * gradi

            # each pass, compute surrogate loss & error rates:
            Jsur.append( self.nll(X,Y) + reg*np.sum(self.theta**2) )
            J01.append( self.err(X,Y) )
            if plot is not None: plot(self,X,Y,Jsur,J01)
            #print Jsur

            # check stopping criteria:
            it += 1
            done = (it > stopIter) or ( (it>1) and (abs(Jsur[-1]-Jsur[-2])<stopTol) )


################################################################################
################################################################################
################################################################################
    def lossLogisticNLL(self, X,Y, reg=0.0):
        M,N = X.shape
        P = self.predictSoft(X)
        J = - np.sum( np.log( P[range(M),Y[:]] ) )   # assumes Y=0...C-1
        Y = ml.to1ofK(Y,self.classes)
        DJ= NotImplemented ##- np.sum( P**Y
        return J,DJ
        

#    def TODOtrain(self, X, Y, reg=0.0, 
#                    initStep=1.0, stopTol=1e-4, stopIter=5000, 
#                    loss=None,batchsize=1,
#                    plot=None):
#        """
#        Train the linear classifier.  
#        """
#        self.theta,X,Y = twod(self.theta), arr(X), arr(Y)   # convert to numpy arrays
#        M,N = X.shape
#        if Y.shape[0] != M:
#            raise ValueError("Y must have the same number of data (rows) as X")
#        self.classes = np.unique(Y)
#        if self.theta.shape[1] != N+1:         # if self.theta is empty, initialize it!
#            self.theta = np.random.randn(1,N+1)
#
#        it   = 0
#        done = False
#        Jsur = []
#        J01  = []
#        while not done:
#            step = (2.0 * initStep) / (2.0 + it)   # common 1/iter step size change
#            for i in range(M):  # for each data point TODO: batchsize
#                _,gradi = loss(self,Xbatch,Ybatch)
#                self.theta = self.theta - step * gradi
#
#            # each pass, compute surrogate loss & error rates:
#            Jsur.append( loss(self,X,Y) )
#            J01.append( self.err(X,Y) )
#            if plot is not None: plot(self,X,Y,Jsur,J01)
#
#            # check stopping criteria:
#            it += 1
#            done = (it > stopIter) or ( (it>1) and (abs(Jsur[-1]-Jsur[-2])<stopTol) )
#
#
################################################################################
################################################################################
################################################################################

