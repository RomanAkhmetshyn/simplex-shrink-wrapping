# simplex-shrink-wrapping
Simplex shrink-wrapping algorithm in Python based on Daniel R. Fuhrmann "Simplex shrink-wrap algorithm"(1999)

### Algorithm

The algorithm is entirely well described in the original work [1]. Briefly, it is a gradient descent method that computes simplex with the smallest volume around (M-1)-dimensional data points. A simplex in
(M-1)-dimensional space is a polyhedron with M vertices (which makes it different from a convex hull): on a line the simplex is a line segment, on a plane the simplex is a triangle, in three dimensions the simplex is a tetrahedron, etc.
The proposed function for the gradient descent H(A,X) consists of the volume function V(A) and a penalty function F(A,X), where X is data points' coordinates represented by [(M-1) x N] matrix (N = number of data points), A is vertices coordinates represented by [(M-1) x M] matrix. However, in the method described in the paper, there are 2 variables for the steps of descent iteration: one is constant and is multiplied by ∇V(A), and the other one decreases with iteration and is multiplied by ∇F(A,X).

### Current limitation
For now, the code works only for 2D data.

### Disclaimer
I'm no professional in gradient descent or linear algebra. I could not find any code in Python that would replicate the algorithm from the paper, so this is my own attempt to "translate" mathematic formulas into Python. The code also does not work smoothly, you need to adjust step variables depending on your data manually and any slight change may ruin the whole descent.


### Reference
[1] - Daniel R. Fuhrmann, "Simplex shrink-wrap algorithm," Proc. SPIE 3718, Automatic Target Recognition IX, (24 August 1999); https://doi.org/10.1117/12.359990
