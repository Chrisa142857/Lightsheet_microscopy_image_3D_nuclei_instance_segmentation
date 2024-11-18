import torch as th
from builtins import super
import numpy as np
import torch as th
import numbers
from warnings import warn
import math


def is_positive_semi_definite(R):
    if not th.is_tensor(R):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a tensor, instead got : {}'.format(R))
    return th.all(th.real(th.linalg.eigvals(R))>0)


def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).
    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = th.sub(X[None, :, :], Y[:, None, :])
    err = th.pow(diff, 2)
    return th.sum(err) / (D * M * N)

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.
    Attributes
    ----------
    X: numpy array
        NxD array of target points.
    Y: numpy array
        MxD array of source points.
    TY: numpy array
        MxD array of transformed source points.
    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.
    N: int
        Number of target points.
    M: int
        Number of source points.
    D: int
        Dimensionality of source and target points
    iteration: int
        The current iteration throughout registration.
    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.
    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.
    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).
    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.
    diff: float (positive)
        The absolute difference between the current and previous objective function values.
    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.
    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.
    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.
    Np: float (positive)
        The sum of all elements in P.
    """

    def __init__(self, X, Y, device, sigma2=None, max_iterations=None, tolerance=None, w=None, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D tensor array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        self.device = th.device(device)
        self.X = th.tensor(X, dtype=th.float32).to(self.device)
        self.Y = th.tensor(Y, dtype=th.float32).to(self.device)
        self.TY = th.tensor(Y, dtype=th.float32).to(self.device)
        self.sigma2 = initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2
        if type(self.sigma2) is not th.Tensor:
            self.sigma2 = th.tensor(self.sigma2, dtype=th.float32).to(self.device)
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = th.tensor(0.5, dtype=th.float32).to(self.device) if tolerance is None else tolerance
        if type(self.tolerance) is not th.Tensor:
            self.tolerance = th.tensor(self.tolerance, dtype=th.float32).to(self.device)
        self.w = th.tensor(0.0, dtype=th.float32).to(self.device) if w is None else w
        if type(self.w) is not th.Tensor:
            self.w = th.tensor(self.w, dtype=th.float32).to(self.device)
        self.max_iterations = 1000 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = th.tensor(np.inf, dtype=th.float32).to(self.device)
        self.q = th.tensor(np.inf, dtype=th.float32).to(self.device)
        self.P = th.zeros((self.M, self.N), dtype=th.float32).to(self.device)
        self.Pt1 = th.zeros((self.N, 1), dtype=th.float32).to(self.device)
        self.P1 = th.zeros((self.M, 1), dtype=th.float32).to(self.device)
        self.PX = th.zeros((self.M, self.D), dtype=th.float32).to(self.device)
        self.Np = th.tensor(0., dtype=th.float32).to(self.device)

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.
        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.
        
        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.
        
        registration_parameters:
            Returned params dependent on registration method used. 
        """
        self.transform_point_cloud()
        Np_zero = False
        while self.iteration < self.max_iterations and self.diff > self.tolerance and not self.t.isnan().any():
            self.pre_t = self.t
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q.detach().cpu().numpy(), 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        if self.t.isnan().any():
            self.t = self.pre_t

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = th.sum(th.pow(self.X[None, :, :] - self.TY[:, None, :], 2), dim=2) # (M, N)
        P = th.exp(th.div(-P, (2.*self.sigma2)))
        c = th.pow(2.*th.tensor(math.pi, dtype=th.float32)*self.sigma2, (self.D/2.))*self.w/(1. - self.w)*self.M/self.N

        den = th.sum(P, dim = 0, keepdims = True) # (1, N)
        den = th.clamp(den, th.finfo(self.X.dtype).eps, None) + c

        self.P = th.div(P, den)
        self.Pt1 = th.sum(self.P, dim=0).reshape(-1, 1)
        self.P1 = th.sum(self.P, dim=1).reshape(-1, 1)
        self.Np = th.sum(self.P1)
        self.PX = th.mm(self.P, self.X)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()

        
class RigidRegistration(EMRegistration):
    """
    Rigid registration.
    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.
    t: numpy array
        1xD initial translation vector.
    s: float (positive)
        scaling parameter.
    A: numpy array
        Utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
    """
    # Additional parameters used in this class, but not inputs.
    # YPY: float
    #     Denominator value used to update the scale factor.
    #     Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    # X_hat: numpy array
    #     Centered target point cloud.
    #     Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.


    def __init__(self, R=None, t=None, s=None, scale=False, rotate=False, force_z0=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            raise ValueError(
                'Rigid registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D))

        if R is not None and ((R.ndim != 2) or (R.shape[0] != self.D) or (R.shape[1] != self.D) or not is_positive_semi_definite(R)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, R))

        if t is not None and ((t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.D)):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))

        if s is not None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError(
                'The scale factor must be a positive number. Instead got: {}.'.format(s))

        self.R = th.eye(self.D, dtype=th.float32).to(self.device) if R is None else R
        if type(self.R) is not th.Tensor:
            self.R = th.tensor(self.R, dtype=th.float32).to(self.device)
        self.t = th.atleast_2d(th.zeros((1, self.D), dtype=th.float32)).to(self.device) if t is None else t
        if type(self.t) is not th.Tensor:
            self.t = th.tensor(self.t, dtype=th.float32).to(self.device)
        self.s = th.tensor(1, dtype=th.float32).to(self.device) if s is None else s
        if type(self.s) is not th.Tensor:
            self.s = th.tensor(self.s, dtype=th.float32).to(self.device)
        self.rotate = rotate
        self.scale = scale
        self.force_z0 = force_z0

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.
        """
        # target point cloud mean
        muX = th.div(th.sum(self.PX, dim=0),self.Np)
        # source point cloud mean
        muY = th.div(th.sum(th.mm(self.P.permute(1, 0), self.Y), dim=0), self.Np)
        self.X_hat = th.sub(self.X, th.tile(muX, (self.N, 1)))
        # centered source point cloud
        Y_hat = th.sub(self.Y, th.tile(muY, (self.M, 1)))
        self.YPY = th.mm(self.P1.permute(1, 0), th.sum(th.mul(Y_hat, Y_hat), dim=1).reshape(-1, 1)).reshape(-1, )

        self.A = th.mm(self.X_hat.permute(1, 0), self.P.permute(1, 0)).to(self.device)
        self.A = th.mm(self.A, Y_hat)
        try:
            # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
            U, _, V = th.linalg.svd(self.A, full_matrices=True)
        except:
            return
        C = th.ones((self.D, )).to(self.device)
        C[self.D-1] = th.linalg.det(th.mm(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.rotate is True:
            self.R = (th.mm(th.mm(U, th.diag(C)), V)).permute(1, 0)
        else:
            self.R = th.eye(self.D, device=self.device)
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.scale is True:
            self.s = th.trace(th.mm(self.A.permute(1, 0), self.R.permute(1, 0))) / self.YPY
        else:
            self.s = 1
        self.t = th.sub(muX.reshape(-1, 1), self.s * th.mm(self.R.permute(1, 0), muY.reshape(-1, 1))).permute(1, 0)
        if self.force_z0 is True: 
            self.t[0, 0] = 0

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.
        Attributes
        ----------
        Y: numpy array
            Point cloud to be transformed - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
        """
        if Y is None:
            self.TY = self.s * th.mm(self.Y, self.R) + self.t
            return
        else:
            return self.s * th.mm(Y, self.R) + self.t

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.q
        trAR = th.trace(th.mm(self.A, self.R))
        xPx = th.mm(self.Pt1.permute(1, 0), th.sum(th.mul(self.X_hat, self.X_hat), dim=1).reshape(-1, 1)).reshape(-1, )
        self.q = (xPx - 2. * self.s * trAR + self.s * self.s * self.YPY) / \
            (2. * self.sigma2) + self.D * self.Np/2. * th.log(self.sigma2)
        self.diff = th.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = (self.tolerance / 10).clone()

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.
        Returns
        -------
        self.s: float
            Current estimate of the scale factor.
        
        self.R: numpy array
            Current estimate of the rotation matrix.
        
        self.t: numpy array
            Current estimate of the translation vector.
        """
        return self.s, self.R, self.t
