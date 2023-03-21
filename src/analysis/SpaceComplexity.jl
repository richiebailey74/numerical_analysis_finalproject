module SpaceComplexity
export ridge_SC, lasso_SC

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

# Actual Ridge Algorithm
function ridge_SC(X,y,lambda)                               # X: (m x n), Y: (m x 1), lambda: (constant)
    n = size(X)[2]                                          # O(1)    1 variable
    Im = 1 * Matrix(I, n, n)                                # O(n^2)  (n x n) matrix
    w = inv(transpose(X)*X + lambda*Im) * transpose(X) * y  # O(n)    (n x 1) matrix
                                                            # O(n^2)  (n x n) matrices [3 temporary matrices]
                                                            # O(m*n)  (n x m) matrix   [1 temporary matrix]
    pred = X * w                                            # O(m)    (m x 1) matrix
    err = y - pred                                          # O(m)    (m x 1) matrix
    
    return w, pred, err                                     # O(1)    Return on Stack
end

# Total Space Complexity
# O(n^2) + O(n^2) + O(n^2) + O(n^2) + O(n*m) + O(n) + O(m) + O(m) + O(1) + O(1)

# Final Space Complexity
# O(n^2) + O(n*m)




# Parameters - X: (m x n), y: (m x 1), lambda: constant, learning_rate: constant, iterations: constant
# Variables  - m: rows, n: cols, i: iterations
function lasso_SC(X, y, lambda)
    learning_rate = .025                    # O(1)   [Variable Assignment]
    iterations = 8000                       # O(1)   [Variable Assignment]
    l1_penalty = lambda                     # O(1)   [Variable Assignment]
    n = size(X)[2]                          # O(1)   [Accessing Size Variable]
    m = size(X)[1]                          # O(1)   [Accessing Size Variable]
    w = zeros(n)                            # O(n)   [Creation of vector /w size n]
    b = 0                                   # O(1)   [Variable Assignment]
    
    for i in 1:iterations                   # O(i)   [Loop]
        y_pred = zeros(m)                   # O(m)       [Creation of vector of size m]
        y_res = y - y_pred                  # O(m)       [Creation of vector of size m]
        
        for k in 1:m                        # O(m)       [Loop]
            y_pred[k] = dot(X[k,:], w) + b  # O(1)           [Storing Value]
                                            # O(m)           [+ Temporary Dot Product Vector]
                                            # O(1)           [+ Temporary Scalar Value]
        end
        
        #calculate gradients
        dW = zeros(n)                       # O(n)       [Creation of vector /w size n]
        for j in 1:n                        # O(n)       [Loop]
            dp_XY = -2 * dot(X[:,j], y_res) # O(m)          [Creation of vector size m]
                                            # O(1)          [+ Temporary Scalar Value]
            if w[j] > 0                     # ----          [Access Index] [IF/ELSE: Space / Iteration = 2*O(1)]
                dW[j] = (dp_XY + l1_penalty) ./ m  # O(1)   [Variable Assignment]
                                                   # O(1)   [+ Temporary Scalar Value]
            else
                dW[j] = (dp_XY - l1_penalty) ./ m  # O(1)   [Variable Assignment]
                                                   # O(1)   [+ Temporary Scalar Value]
            end
        end
        
        db = - 2 * sum(y - y_pred) ./ m            # O(1)   [Variable Assignment]
                                                   # O(1)   [+ Temporary Scalar Value] x2
        
        w = w - learning_rate*dW                   # O(1)   [Variable Assignment]
                                                   # O(1)   [+ Temporary Scalar Value]
        
        b = b - learning_rate*db                   # O(1)   [Variable Assignment]
                                                   # O(1)   [+ Temporary Scalar Value]
    end
    
    return w, b                                    # O(n)   [Returning (n x 1)]
                                                   # O(1)   [+ Returning scalar]
    
end

# Total Space Complexity
# Variables: m: rows, n: columns, i: iterations

# (6*O(1) + O(n)) + i*(O(m) + O(m) + m*(O(1) + O(m) + O(1)) + O(n) + n*(O(m) + 3*O(1)) + O(n) + 7*O(1))
# (6*O(1) + O(n)) + i*(O(m^2) + O(m*n) + 5*O(n) + 4*O(m) + 7*O(1))
# (6*O(1) + O(n)) + i*(O(m^2) + O(m*n) + 5*O(n) + 4*O(m) + 7*O(1))
# O(m^2*i) + O(m*n*i) + 5*O(n*i) + 4*O(m*i) + 7*O(i) + O(n) + 6*O(1)
# O(m^2*i) + O(m*n*i) + O(n*i) + O(m*i) + O(i) + O(n) + O(1)

# Final Space Complexity 
# O(m^2*i) + O(m*n*i)



# Space Complexity References
# -------------------------------
# [Space Complexity Calculates Temp Vars?] https://www.studytonight.com/data-structures/space-complexity-of-algorithms
