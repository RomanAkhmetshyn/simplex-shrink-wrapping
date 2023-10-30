import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import trange


def obj_func(A,X, alpha):
    '''
    Objective function of the gradient descent. 
    It has no use in this code and is just for information.

    Parameters
    ----------
    A : array
        Simplex vertices coordinates represented by [(M-1) x M] matrix. M-1 is
        dimensionality of the data.
    X : array
        Data points' coordinates represented by [(M-1) x N] matrix (N = number of data points).
    alpha : float
        A multiplier to balance the effects of the volume term and the penalty term.

    Returns
    -------
    float
        Result of the objective function.

    '''
    return volume(A)+alpha*(1/np.sum(penalty(A,X)))
    
    
def volume(A):
    '''
    Calculate volume (area) of the simplex using coordinates of vertices.

    Parameters
    ----------
    A : array
        Simplex vertices coordinates represented by [(M-1) x M] matrix. M-1 is
        dimensionality of the data.

    Returns
    -------
    volume_A : float
        Volume (area) of the simplex.

    '''
    M = A.shape[1]  # Dimension of the simplex
    volume_A=0

    for i in range(M):
        a_i = A[:, i]  # Extract the i-th column as a_i
        a_i = a_i[:, np.newaxis]
        ones = np.ones(M - 1) # Create a row vector of ones of appropriate size
        
        ones = ones.reshape(1, 2)

        
        A_i = np.delete(A, i, axis=1)  # Create A_i by removing the i-th column
        
        A_i_minus_a_i_1T = A_i - np.dot(a_i, ones)  # Subtract a_i * 1^T from A_i

        det_A_i_minus_a_i_1T = np.linalg.det(A_i_minus_a_i_1T)  # Calculate the determinant of A_i - a_i * 1^T

        volume_A += 1 / math.factorial(M - 1) * det_A_i_minus_a_i_1T
    
    return volume_A

def penalty(A,X):
    '''
    
    Penalty function to enforce the constraint that all the
    data lie in the interior of the simplex defined by A.

    Parameters
    ----------
    A : array
        Simplex vertices coordinates represented by [(M-1) x M] matrix. M-1 is
        dimensionality of the data.

    X : array
        Data points' coordinates represented by [(M-1) x N] matrix (N = number of data points).

    Returns
    -------
    float
        Penalty function matrix (P). P is a solution to square system of linear equations X=AP.
        If simplex is not overlaping data points, all elements of P will be between 0 and 1.

    '''
    
    ones_row = np.ones((1, X.shape[1]))
    tildX=np.vstack((X, ones_row))

    
    ones_row = np.ones((1, A.shape[1]))
    tildA=np.vstack((A, ones_row))

    
    
    P = np.linalg.lstsq(tildA, tildX, rcond=None)[0]
    
    
    
    return P#, 1/np.sum(P)

def compute_auxiliaries(A):
    '''
    Calculate matrix B consisting of vectors b. Used for gradient of volume function.
    Vectors b_i are normal from the affine plane determined by the remaining vertices
    to the a_i vertice. 

    Parameters
    ----------
    A : array
        Simplex vertices coordinates represented by [(M-1) x M] matrix. M-1 is
        dimensionality of the data.

    Raises
    ------
    ValueError
        If data is not 2D.

    Returns
    -------
    array
        [(M-1) x M] Matrix of normals.

    '''
    M = A.shape[1]
    b_vectors = []
    if M==3: #2D case
    
        for i in range(M):
            # Select the i-th vertex
            a_i = A[:, i]
            
            # Create a matrix A_i by removing the i-th vertex from A
            A_i = np.delete(A, i, axis=1)
            
            
            
            a_i_proj=find_projection(A_i[:, 0], A_i[:, 1], a_i)
            
            
            normal=a_i-a_i_proj
            
            # vector= A_i[:, 0]-A_i[:, 1]
            
            # print(round(np.dot(normal,vector)))
            
            b_vectors.append(normal)
            
        return np.array(b_vectors).T
            
            
    else:
        raise ValueError("Dimention is not 2D...")

    

def grad_volume(volume, A):
    '''
    Gradient of the volume function.

    Parameters
    ----------
    volume : float
        Volume of the simplex defined by A vertices.
    A : array
        Simplex vertices coordinates represented by [(M-1) x M] matrix. M-1 is
        dimensionality of the data.

    Returns
    -------
    array
        [(M — 1) x M] matrix.

    '''
    
    B=compute_auxiliaries(A)
    
    squared_magnitudes = np.sum(B**2, axis=0)

    # Divide each vector coordinate by its squared magnitude
    normalized_B = B / squared_magnitudes
    
    grad=volume*normalized_B
    
    return grad

def grad_penalty(A, P):
    '''
    Gradient of the penalty function.

    Parameters
    ----------
    A : array
        Simplex vertices coordinates represented by [(M-1) x M] matrix. M-1 is
        dimensionality of the data.
    P : array
        Penalty function [M X N] matrix. M is dimensionality of data +1, 
        N is number of data points. 

    Returns
    -------
    array
        [(M — 1) x M] matrix.

    '''
    ones_row = np.ones((1, A.shape[1]))
    tildA=np.vstack((A, ones_row))
    
    D = 1 / (P**2)

    grad=np.linalg.inv(tildA).T @ D @ P.T
    
    return np.delete(grad, -1, axis=0)

def find_projection(point1, point2, point_i):
    '''
    Calculate coordinates for the projection of a_i vertice onto
    the linie created by other vertices (2D case) in order to calculate
    normal vector from projection to a_i.

    Parameters
    ----------
    point1 : array
        Coordinates of the first vertice as a column vector.
    point2 : array
        Coordinates of the second vertice as a column vector.
    point_i : array
        Coordinates of the remaining vertice as a column vector.

    Returns
    -------
    array
        Coordinates of the projected vertice of point_i as a column vector.

    '''
    # Extract the coordinates of the first two points
    x1, y1 = point1
    x2, y2 = point2

    # Check if the line is horizontal (slope is 0)
    if y2 - y1 == 0:
        # Projection point has the same y-coordinate as the line
        xp = point_i[0]  # Projection point x-coordinate is the same as the x-coordinate of the third point
        yp = y1
    else:
        # Calculate the slope and intercept of the line through the first two points
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # Calculate the perpendicular slope
        m_ortho = -1 / m

        # Extract the coordinates of the third point
        xi, yi = point_i

        # Calculate the y-intercept of the line through the third point
        b_ortho = yi - m_ortho * xi

        # Calculate the x-coordinate of the projection point
        xp = (b - b_ortho) / (m_ortho - m)

        # Calculate the y-coordinate of the projection point
        yp = m * xp + b

    return np.array([xp, yp])

def new_params(old_A, X, mu1, mu2, P, volume):
    '''
    Calculate new coordinates of vertices for gradient descent.

    Parameters
    ----------
    old_A : array
        Previous value of matrix A.
    X : array
        Data points' coordinates represented by [(M-1) x N] matrix (N = number of data points).
    mu1 : float
        Step variable for the gradient of Volume function.
    mu2 : TYPE
        Step variable for the Penalty function.
    P : array
        Penalty function [M X N] matrix. M is dimensionality of data +1, 
        N is number of data points. 
    volume : float
        Volume of the simplex defined by A vertices.

    Returns
    -------
    A : array
        New value of A.

    '''
    gradV=grad_volume(volume, old_A)
    gradF=grad_penalty(old_A, P)

    
    A=old_A - mu1 * gradV - mu2 * gradF
    # A=old_A-alpha*(gradV+gradF)
    
    
    return A

def shrink_wrap(X, mu1, init_mu2, epochs, A=None, plot_prog=None):
    
    if A==None:
        centroid = np.mean(X, axis=1)

        max_distance = np.max(np.linalg.norm(X - centroid[:, np.newaxis], axis=0))

        init_a = centroid + np.array([-1, -1]) * max_distance * 2
        init_b = centroid + np.array([1, -1]) * max_distance * 2
        init_c = centroid + np.array([0, 2]) * max_distance * 2

        A=np.vstack((init_a, init_b, init_c))
        A=A.T
        
    P=penalty(A, X)
    if np.any(P > 1):
        raise ValueError("Initial simplex is overlaping with data. Change coordinates of A...")
    
    V=volume(A)
    
    if plot_prog:

        for epoch in trange(epochs):

            mu2=0.000001 * (epochs+1 - epoch)/(epochs)

            A=new_params(A, X, mu1, mu2, P, V)
            P=penalty(A, X)
            V=volume(A)
            
            if epoch%plot_prog==0:
                
                a,b,c=A.T
                
                plt.scatter(X[0], X[1], s=0.5)
                plt.plot([a[0], b[0], c[0], a[0]],
                          [a[1], b[1], c[1], a[1]], 'k-', linewidth=0.2)
                plt.title(f'simplex at every {plot_prog} iteration')

                
                
            if V<0:
                raise ValueError("Negative volume. Change step parameters...")
                break
            
        plt.show()   
        print('final area ',V)
        return A
    
    else:
        
        for epoch in trange(epochs):
            
            mu2=0.000001 * (epochs+1 - epoch)/(epochs)
            
            A=new_params(A, X, mu1, mu2, P, V)
            P=penalty(A, X)
            V=volume(A)
            
            if V<0:
                raise ValueError("Negative volume. Change step parameters...")
                break
        
        print('final area ',V)
        return A



N = 10  #  number of data points
M = 3  #  dimensionality M-1


X = np.loadtxt('D:\SIMP0136-variabilite\spectra-20231003T143252Z-001\pcs.txt').T

epochs=10000
mu1=0.001
init_mu2=0.000001

A=shrink_wrap(X, mu1, init_mu2, epochs, plot_prog=100)
V=volume(A)
    
a,b,c=A.T
fig = plt.figure(figsize=(6, 6))
plt.scatter(X[0], X[1], s=0.5)
plt.plot([a[0], b[0], c[0], a[0]],
          [a[1], b[1], c[1], a[1]], 'k-', linewidth=0.8)
plt.title(f'{epochs} epochs, Area={V:.3f}')
plt.xlim(-0.5,0.9)
plt.ylim(-0.5, 0.9)
plt.show()
    
