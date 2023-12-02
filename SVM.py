"""Support-vector-machine (SVM) model for supervised machine learning.

Notes
-----
  This script is version v0. It provides the base for all subsequent
  iterations of the project.

Requirements
------------
  See "requirements.txt"

"""

#%% import libraries and modules
import os
import numpy as np  
from scipy import optimize
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.inspection import DecisionBoundaryDisplay

#%% figure parameters
plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['font.size']= 15
plt.rcParams['lines.linewidth'] = 2

#%% build SVM class

class SVM:
    """SVM class."""
    
    def __init__(self, class_size=100, C=1.0, kernel='rbf'):
        self.class_size = class_size
        self.C = C
        self.kernel = kernel
        
    def make_inputs(self):
        """Create input patterns."""
        if self.kernel == 'linear':
            return self.make_blobs()
        if self.kernel == 'rbf':
            return self.make_double_moon()
        
    def make_blobs(self):
        """Create linearly separable data."""
        # set random seed
        np.random.seed(42)
        # generate class A and class B scatter
        class_a_inputs = np.random.randn(self.class_size, 2) + np.array([0, 4])
        class_b_inputs = np.random.randn(self.class_size, 2) + np.array([4, 0])
        # concatenate inputs
        X = np.vstack((class_a_inputs, class_b_inputs))
        return X
        
    def make_double_moon(self, radius=1.0, thickness=0.2, separation=-0.2):
        """Create non-linearly separable data."""
        # set random seed
        np.random.seed(42)
        # specify center
        center_a = np.array([ (radius + thickness)/2, -separation/2])
        center_b = np.array([-(radius + thickness)/2,  separation/2])
        # specify radius
        radius_a = np.random.rand(self.class_size) * thickness + radius
        radius_b = np.random.rand(self.class_size) * thickness + radius
        # specify angle
        angle_a  = np.random.rand(self.class_size) * np.pi
        angle_b  = np.random.rand(self.class_size) * np.pi + np.pi
        # specify points
        point_a = np.array((radius_a * np.cos(angle_a), radius_a * np.sin(angle_a))).T
        point_b = np.array((radius_b * np.cos(angle_b), radius_b * np.sin(angle_b))).T
        # generate class A and class B scatter
        class_a_inputs = point_a - center_a
        class_b_inputs = point_b - center_b
        # concatenate inputs
        X = np.vstack((class_a_inputs, class_b_inputs))
        return X
        
    def normalize_inputs(self, X):
        """Normalize input features between -1 and +1."""
        X_scaled = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1 
        return X_scaled
        
    def make_targets(self):
        """Create target patterns."""
        # set class A targets to +1
        class_a_targets = np.zeros(self.class_size) + 1
        # set class B targets to -1
        class_b_targets = np.zeros(self.class_size) - 1
        # concatenate targets
        y = np.hstack((class_a_targets, class_b_targets))
        return y
        
    def lagrange_dual(self, alpha, X, y):
        """Define the objective function for the dual problem of constrained optimization."""
        term_one = np.dot(alpha[:, np.newaxis], alpha[:, np.newaxis].T)
        term_two = np.dot(y[:, np.newaxis], y[:, np.newaxis].T)
        if self.kernel == 'linear':
            term_three = self.linear_kernel(X)
        if self.kernel == 'rbf':
            term_three = self.rbf_kernel(X)
        result = 0.5 * sum(sum(term_one * term_two * term_three)) - sum(alpha)
        return result
        
    def lagrange_alpha(self, alpha, X, y):
        """Define the partial derivative of lagrange dual w.r.t alpha."""
        term_one = alpha
        term_two = np.dot(y[:, np.newaxis], y[:, np.newaxis].T)
        if self.kernel == 'linear':
            term_three = self.linear_kernel(X)
        if self.kernel == 'rbf':
            term_three = self.rbf_kernel(X)
        result = np.dot(term_one, term_two * term_three) - np.ones_like(term_one)
        return result
        
    def optimize_alpha(self, X, y):
        """Optimize lagrange multipliers - alpha."""
        # initialize alpha
        alpha0 = np.zeros(len(X))
        # set lower and upper bounds on alpha
        lb = np.full(len(X), 0)
        ub = np.full(len(X), self.C)
        # identity matrix
        I = np.eye(len(X))
        # specify constraints
        constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                       {'type': 'ineq', 'fun': lambda a: np.dot(I, a), 'jac': lambda a: I})
        # minimize the objective function
        alpha = optimize.minimize(fun=self.lagrange_dual, 
                                  x0=alpha0,
                                  args=(X, y),
                                  method='SLSQP',
                                  jac=self.lagrange_alpha,
                                  constraints=constraints,
                                  bounds=zip(lb, ub))
        return alpha.x
        
    def get_optimum_variables(self, X, y, alpha, epsilon=1e-6):
        """Get support vectors, optimum weight and bias."""
        # get support vectors
        support_vectors = X[(alpha >= epsilon) & (alpha <= self.C)]
        # compute weight
        weight = np.sum(alpha * y * X.T, axis=1)
        # compute bias
        bias = np.mean(y[(alpha >= epsilon) & (alpha <= self.C)] - np.dot(weight, support_vectors.T))
        
        return support_vectors, weight, bias
        
    def get_hyperplane(self, X, y, alpha, weight, bias):
        """Get optimum hyperplane that separates the two classes."""
        # create mesh for plotting
        num_pts = 1000
        x_min, y_min = X.min(axis=0)
        x_max, y_max = X.max(axis=0)
        xx, yy = np.meshgrid(np.linspace(x_min - 1, x_max + 1, num_pts),
                             np.linspace(y_min - 1, y_max + 1, num_pts))
        
        # build test data
        X_test = np.c_[xx.ravel(), yy.ravel()]
        
        # apply decision function based on specified kernel
        if self.kernel == 'linear':
            # linear kernel
            result = np.dot(X_test, X.T)
            # decision function
            zz     = np.sign(np.sum(alpha * y * result, axis=1) + bias)
            zz_pos = np.sign(np.sum(alpha * y * result, axis=1) + bias - 1)
            zz_neg = np.sign(np.sum(alpha * y * result, axis=1) + bias + 1)
            
        if self.kernel == 'rbf':
            # radial basis function (rbf) kernel
            gamma = 1/(2 * X.var())
            result = np.exp(-gamma * np.sum(np.square(X_test[:,:,np.newaxis] - X[:,:,np.newaxis].T), axis=1))
            # decision function
            zz     = np.sign(np.sum(alpha * y * result, axis=1))
            zz_pos = np.sign(np.sum(alpha * y * result, axis=1) - 1)
            zz_neg = np.sign(np.sum(alpha * y * result, axis=1) + 1)

        # reshape decision response for contour plot
        zz     = zz.reshape(xx.shape)
        zz_pos = zz_pos.reshape(xx.shape)
        zz_neg = zz_neg.reshape(xx.shape)
        return xx, yy, zz, zz_pos, zz_neg
    
    def get_classification_performance(self, X, y, alpha, bias):
        """Compute classification performance using mean squared error."""
        if self.kernel == 'linear':
            obtained = np.sign(np.sum(alpha * y * self.linear_kernel(X), axis=1) + bias)
        if self.kernel == 'rbf':
            obtained = np.sign(np.sum(alpha * y * self.rbf_kernel(X), axis=1))
        actual = y
        mse = 1/len(X) * sum((obtained - actual)**2)
        classification_performance = (1 - mse) * 100
        return classification_performance
    
    def rbf_kernel(self, X):
        """Apply radial basis function kernel."""
        gamma = 1/(2 * X.var())
        result = np.exp(-gamma * np.sum(np.square(X[:,:,np.newaxis] - X[:,:,np.newaxis].T), axis=1))
        return result
    
    def linear_kernel(self, X):
        """Apply linear kernel."""
        result = np.dot(X, X.T)
        return result
        
    def plot_svm(self, X, y, alpha, support_vectors, bias, xx, yy, zz, zz_pos, zz_neg):
        """Plot svm classification."""
        # specify class A and class B from input patterns
        class_a = X[:self.class_size, :]
        class_b = X[self.class_size:, :]
        
        # get classification performance
        classification_performance = self.get_classification_performance(X, y, alpha, bias)
        
        cwd = os.getcwd()                                                       # get current working directory
        fileName = 'images'                                                     # specify filename

        # filepath and directory specifications
        if os.path.exists(os.path.join(cwd, fileName)) == False:                # if path does not exist
            os.makedirs(fileName)                                               # create directory with specified filename
            os.chdir(os.path.join(cwd, fileName))                               # change cwd to the given path
            cwd = os.getcwd()                                                   # get current working directory
        else:
            os.chdir(os.path.join(cwd, fileName))                               # change cwd to the given path
            cwd = os.getcwd()                                                   # get current working directory
        
        # plot numpy svm
        fig, ax = plt.subplots()
        # plot data and support vectors
        h_class_a = ax.scatter(class_a[:, 0], class_a[:, 1], s=50, color='r', marker='o', label='class A')
        h_class_b = ax.scatter(class_b[:, 0], class_b[:, 1], s=50, color='b', marker='x', label='class B')
        h_support_vectors = ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                                       s=100, color='k', facecolors='none',
                                       label='support vector', linewidth=2)
        # plot optimal hyperplane, (+) margin and (-) margin
        cs     = ax.contour(xx, yy, zz,     colors='k', linestyles='-')
        cs_pos = ax.contour(xx, yy, zz_pos, colors='k', linestyles='--')
        cs_neg = ax.contour(xx, yy, zz_neg, colors='k', linestyles='-.')
        ax.legend([h_class_a, h_class_b, h_support_vectors,
                   cs.legend_elements()[0][0], cs_pos.legend_elements()[0][0], cs_neg.legend_elements()[0][0]],
                  ['class A', 'class B', 'support vector',
                   'optimal hyperplane', '(+) margin', '(-) margin'], loc='upper right', ncol=2)
        ax.set_title(f'numpy - classification performance = {classification_performance:.0f}%')
        fig.savefig(os.path.join(os.getcwd(), 'figure_1'))
        
        # plot sklearn svm 
        clf = svm.SVC(kernel=self.kernel, C=self.C)
        clf.fit(X, y)

        fig, ax = plt.subplots()
        # plot data and support vectors
        ax.scatter(class_a[:, 0], class_a[:, 1], s=50, color='r', marker='o', label='class A')
        ax.scatter(class_b[:, 0], class_b[:, 1], s=50, color='b', marker='x', label='class B')
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=100, linewidth=2, facecolors='none', edgecolors='k', label='support vector')
        # plot optimal hyperplane, (+) margin and (-) margin
        DecisionBoundaryDisplay.from_estimator(clf, X, plot_method='contour',
                                               colors='k', levels=[-1, 0, 1], 
                                               alpha=0.5, linestyles=['-.', '-', '--'],
                                               ax=ax)
        ax.set_title(f'sklearn - classification performance = {(1 - mean_squared_error(y, clf.predict(X)))*100:.0f}%')
        ax.legend()
        fig.savefig(os.path.join(os.getcwd(), 'figure_2'))

#%% instantiate SVM class

model = SVM()

#%% create input and target patterns

X = model.make_inputs()
X_scaled = model.normalize_inputs(X)
y = model.make_targets()

#%% optimize lagrange multipliers
        
alpha = model.optimize_alpha(X_scaled, y)

#%% find support vectors, weight and bias

support_vectors, weight, bias = model.get_optimum_variables(X_scaled, y, alpha)

#%% get optimal hyperplane, (+) margin and (-) margin

xx, yy, zz, zz_pos, zz_neg = model.get_hyperplane(X_scaled, y, alpha, weight, bias)

#%% plot figures

model.plot_svm(X_scaled, y, alpha, support_vectors, bias, xx, yy, zz, zz_pos, zz_neg)
