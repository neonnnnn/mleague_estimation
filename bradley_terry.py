from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import safe_sparse_dot


def convert_feature_matrix(X, n_players):
    n_samples = X.shape[0]
    data = np.tile((4 * np.eye(4) - np.ones((4, 4))).ravel(), n_samples)
    indices = np.array(np.tile(X, 4).ravel())
    indptr = np.arange(0, (n_samples)*4*4+1, 4)
    return csr_matrix((data, indices, indptr), shape=(4*n_samples, n_players))
    
    
class BT(BaseEstimator, RegressorMixin):
    """Bradley-Terry Model for Mahjong.
    
    Parameters
    ----------
    alpha : float, default: 1
        Regularization strength.
    
    Attributes
    ----------    
    self.ridge_ : Ridge instance
        The learned Ridge instance. It is learned for computing (learning)
        self.coef_, which is a reference of self.ridge_.coef_.
        
    self.coef_ : array, shape = [n_players]
        The learned strength vector.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def predict(self, X):
        X = self.enc_.transform(X.ravel()).reshape(X.shape)
        X_feature = convert_feature_matrix(X, len(self.enc_.classes_))
        y_pred = safe_sparse_dot(X_feature, self.coef_, dense_output=True)
        return y_pred.reshape(-1, 4)
    
    def fit(self, X, y):
        """Fit Bradley-Terry model for Mahjong.
        
        Parameters
        ----------
        X : array, shape = [n_samples, 4]
            Training vectors, where n_samples is the number of samples.
            
        y : array- of integer, shape = [n_samples, 4]
            Target values.
            
        Returns
        -------
        self : Estimator
            Returns self.
        """
        if X.shape != y.shape or X.shape[1] != 4 or y.shape[1] != 4:
            msg = "Invalid shape. X.shape and y.shape must be (n_samples, 4)"
            raise ValueError(
                f"{msg}, but X.shape={X.shape}, y.shape={y.shape}"
            )
        n_samples = X.shape[0]
        n_players = len(np.unique(X))
        
        # init encoder
        self.enc_ = LabelEncoder()
        self.enc_.fit(X.ravel())
        
        # create some vectors and matrices for optimization procedure
        X = self.enc_.transform(X.ravel()).reshape(-1, 4)
        X_feature = convert_feature_matrix(X, n_players)
        
        # init and fit Ridge
        alpha = n_samples * self.alpha
        self.ridge_ = Ridge(alpha=alpha, fit_intercept=False, normalize=False)
        self.ridge_.fit(X_feature, y.ravel())
        self.coef_ = self.ridge_.coef_

        return self