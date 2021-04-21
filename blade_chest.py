import numpy as np
from bradley_terry import convert_feature_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from numpy.linalg import solve, inv
from sklearn.utils import check_random_state
from scipy.sparse import csc_matrix
from sklearn.utils.extmath import safe_sparse_dot


class BladeChestInner(BaseEstimator, RegressorMixin):
    """Blade-chest inner model for Mahjong.
    
    Parameters
    ----------
    n_components : int, default: 30
        Rank hyperparameter.
        
    fit_linear: bool, default: True
        Whether to fit the strength weight vector coef_.

    max_iter : int, default: 100
        Maximum number of passes over the dataset to perform.
    
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.
    
    alpha : float, default: 1
        Regularization strength for strength parameters.
    
    beta : float, default: 1
        Regularization strength for blade and chest parameters.
    
    tol : float, default: 1e-6
        Tolerance for the stopping condition.
    
    verbose : boolean, optional, default: False
        Whether to print debugging information.
        
    n_calls : int
        Frequency with which the values of loss and regularizain terms
        are shown.
    
    Attributes
    ----------    
    self.ridge_ : Ridge instance
        The learned Ridge instance. It is learned for computing (learning)
        self.coef_, which is a reference of self.ridge_.coef_.
        
    self.coef_ : array, shape = [n_players]
        The learned strength vector.
    
    self.V_ : array, shape = [n_players, n_components]
        The learned blade vectors.

    self.U_ : array, shape = [n_players, n_components]
        The learned chest vectors.
    """
    
    def __init__(self, n_components=30, fit_linear=True, max_iter=100,
                 random_state=0,alpha=1.0, beta=1.0, tol=1e-4,
                 n_calls=10, verbose=True):
        self.n_components = n_components
        self.fit_linear = fit_linear
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.n_calls = 10
        self.verbose = verbose
        
    def _one_hot(self, X):
        row_ind = np.repeat(np.arange(len(X)), 4)
        data = np.ones(np.prod(X.shape))
        return csc_matrix((data, (row_ind, X.ravel())))
        
    def _predict_blade_chest(self, X, X_one_hot):
        n_samples = X.shape[0]
        n_players = len(self.enc_.classes_)
        y_pred = np.zeros((n_samples, 4))
        V_sum = safe_sparse_dot(X_one_hot, self.V_, dense_output=True)
        U_sum = safe_sparse_dot(X_one_hot, self.U_, dense_output=True)
        for i in range(4):
            indices = X[:, i]
            U_ = self.U_[indices]
            V_ = self.V_[indices]
            y_pred[:, i] += np.sum(V_ * (U_sum - U_), axis=1)
            y_pred[:, i] -= np.sum(U_ * (V_sum - V_), axis=1)
        return y_pred
    
    def predict(self, X):
        X = self.enc_.transform(X.ravel()).reshape(X.shape)
        X_one_hot = self._one_hot(X)
        
        y_pred = self._predict_blade_chest(X, X_one_hot) 
        if self.fit_linear:
            X_feature = convert_feature_matrix(X, len(self.enc_.classes_))
            y_linear = safe_sparse_dot(X_feature, self.coef_, dense_output=True)
            y_pred += y_linear.reshape(-1, 4)
        return y_pred
    
    def _compute_feature_matrix(self, X, P):
        # Compute a feature matrix Z such that 
        # f(X) = np.dot(Z, P.ravel()) + linear_term. P is U_ or V_.
        # TODO: optimize implementation
        n_samples = X.shape[0]
        n_players = len(self.enc_.classes_)
        Z = np.zeros((n_samples*4, n_players*self.n_components))
        for n in range(n_samples):
            for i in range(4):
                start_i = X[n, i] * self.n_components
                stop_i = start_i + self.n_components
                for j in range(4):
                    start_j = X[n, j] * self.n_components
                    stop_j = start_j + self.n_components
                    Z[4*n+i, start_i:stop_i] += P[X[n, j]]
                    Z[4*n+i, start_j:stop_j] -= P[X[n, i]]
        return Z
    
    def _compute_feature_matrix_row(self, X, P, j):
        Z = np.zeros((len(X)*4, self.n_components))
        sum_P = np.zeros((len(X), self.n_components))
        arange = np.arange(len(X))*4
        for i in range(4):
            Z[arange+i] -= P[X[:,  i]]
            sum_P += P[X[:, i]]
        Z[X.ravel() == j] += sum_P
        return Z
        
    def _compute_loss_reg(self, X, X_one_hot, X_feature, y):
        y_pred = self._predict_blade_chest(X, X_one_hot)
        if self.fit_linear:
            y_linear = safe_sparse_dot(X_feature, self.coef_, dense_output=True)
            y_pred += y_linear.reshape(-1, 4)
        loss = 0.5 * np.mean((y_pred-y)**2)
        reg = self.alpha * np.sum(self.coef_**2)
        reg += self.beta * np.sum(self.U_**2 + self.V_**2)
        return loss, reg
    
    def fit(self, X, y):
        """Fit blade-chest inner model for Mahjong
        by alternative least squares.
        
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
                    
        alpha = n_samples * self.alpha
        beta = n_samples * self.beta
        
        # initialize parameters
        random_state = check_random_state(self.random_state)
        self.V_ = random_state.normal(0, 1.0, (n_players, self.n_components))
        self.U_ = random_state.normal(0, 1.0, (n_players, self.n_components))
        self.coef_ = np.zeros(n_players)
        
        # init encoder
        self.enc_ = LabelEncoder()
        self.enc_.fit(X.ravel())
        
        # create some vectors and matrices used in optimization procedure
        X = self.enc_.transform(X.ravel()).reshape(-1, 4)
        X_one_hot = self._one_hot(X)
        y_residual = np.array(y).ravel()
        X_feature = None
        if self.fit_linear:
            X_feature = convert_feature_matrix(X, n_players)
            XTX = safe_sparse_dot(X_feature.T, X_feature, dense_output=True)
            XTX_inv = inv(XTX+ np.eye(n_players)*alpha)
        
        # compute initial value of objective function and start optimization
        loss_old, reg_old = np.inf, np.inf
        y_pred = self._predict_blade_chest(X, X_one_hot)
        for it in range(self.max_iter):
            # optimize linear term
            if self.fit_linear:
                y_linear = safe_sparse_dot(X_feature, self.coef_, dense_output=True)
                y_pred -= y_linear.reshape(-1, 4)
                y_residual = (y - y_pred).ravel()
                XTy = safe_sparse_dot(X_feature.T, y_residual, dense_output=True)
                self.coef_ = np.dot(XTX_inv, XTy)
                y_linear = safe_sparse_dot(X_feature, self.coef_)
                y_pred += y_linear.reshape(-1, 4)
                
            # optimize U and V
            for j in random_state.permutation(n_players):
                start = X_one_hot.indptr[j]
                stop = X_one_hot.indptr[j+1]
                indices = X_one_hot.indices[start:stop]
                
                # optimize V
                Z = self._compute_feature_matrix_row(X[indices], self.U_, j)
                y_pred[indices] -= np.dot(Z, self.V_[j]).reshape(-1, 4)
                y_residual = y[indices] - y_pred[indices]
                ridge = Ridge(alpha=beta).fit(Z, y_residual.ravel())
                self.V_[j] = np.array(ridge.coef_)
                y_pred[indices] += np.dot(Z, self.V_[j]).reshape(-1, 4)
                
                # optimize U
                Z = -self._compute_feature_matrix_row(X[indices], self.V_, j)
                y_pred[indices] -= np.dot(Z, self.U_[j]).reshape(-1, 4)
                y_residual = y[indices] - y_pred[indices]
                ridge = Ridge(alpha=beta).fit(Z, y_residual.ravel())
                self.U_[j] = np.array(ridge.coef_)
                y_pred[indices] += np.dot(Z, self.U_[j]).reshape(-1, 4)
            
            
            # stopping criterion
            loss, reg = self._compute_loss_reg(X, X_one_hot, X_feature, y)
            if self.verbose and it % self.n_calls == 0:
                print(f"Epoch: {it+1} Loss: {loss:.3f} Regularization: {reg:.3f}")
            if (loss+reg) > (loss_old + reg_old - self.tol):
                if self.verbose:
                    print(f"Converged at iteration {it+1}.")
                break
            loss_old, reg_old = loss, reg
        
        return self
    
    # Methods for Testing 
    """
    def _predict_test(self, X):
        X = self.enc_.transform(X.ravel()).reshape(X.shape)
        X_one_hot = self._one_hot(X)
        Z = self._compute_feature_matrix(X, self.U_)
        y_pred = np.dot(Z, self.V_.ravel()).reshape(-1, 4)
        if self.fit_linear:
            X_feature = convert_feature_matrix(X, len(self.enc_.classes_))
            y_pred += np.dot(X_feature, self.coef_).reshape(-1, 4)
        return y_pred
    
    def _predict_test2(self, X):
        X = self.enc_.transform(X.ravel()).reshape(X.shape)
        X_one_hot = self._one_hot(X)
        Z = self._compute_feature_matrix(X, self.V_)
        y_pred = np.dot(Z, self.U_.ravel()).reshape(-1, 4)
        if self.fit_linear:
            X_feature = convert_feature_matrix(X, len(self.enc_.classes_))
            y_pred += np.dot(X_feature, self.coef_).reshape(-1, 4)
        return y_pred
    """