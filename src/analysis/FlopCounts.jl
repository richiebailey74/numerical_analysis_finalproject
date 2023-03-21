module FlopCounts
export ridge_FC, lasso_FC

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

# Ridge Regression Function Decomposed:
# Parameters- X: (m x n), y: (m x 1), lambda: constant
function ridge_FC(X,y,lambda)
    n  = size(X)[2]                          # 0
    Im = 1 * Matrix(I, n, n)                 # n^2          [Constructing (n x n) I Matrix]
    xT = transpose(X)                        # 0            [(m x n)] => (n x m)
    xT_X = xT * X                            # m^2*n        [(n x m) * (m x n)] => (n x n)
    l_Im = lambda * Im                       # n^2          [Scalar multiplication of (n x n) matrix]
    xT_lambda = l_Im + xT_X                  # n^2          [Scalar addition of (n x n) matrix]
    inverse = inv(xT_lambda)                 # n^3 (approx) [Gauss Elimination with matrix of size (n x n)]
    w = inverse * xT                         # n^2*m        [(n x n) * (n x m)] => (n x m)
    w *= y                                   # m*n          [(n x m) * (m x 1)] => (n x 1)
    pred = X * w                             # m*n          [(m x n) * (n x 1)] => (m x 1)
    err = y - pred                           # m            [(m x 1) - (m x 1)] => (m x 1)
    
    return w, pred, err                      # 0
end

# Flop Count
# n^2 + (m^2*n) + n^2 + n^2 + n^3 + n^2*m + m*n + m*n + m
# Total Flop Count = n^3 + (3+m)n^2 + n*m^2 + 2m*n + m

# Note: In the worst case, Gaussian Elimination for n x n will take ((5/6)n^3+(3/2)n^2-(7/6)n) floating point operations.
# Worst Case Flops = (5/6)n^3 + ((9/2)+m)n^2 + n*m^2 + ((5/6)m)n + m




# Lasso Regression Function Decomposed:
# Parameters - X: (m x n), y: (m x 1), lambda: constant, learning_rate: constant, iterations: constant
# Variables  - m: rows, n: cols, i: iterations
# Note: This is the "expanded" version of our function, allowing each instruction to have its own line.
function lasso_FC(X, y, lambda)
    learning_rate = .025                    # 0
    iterations = 8000                       # 0
    l1_penalty = lambda                     # 0
    n = size(X)[2]                          # 0
    m = size(X)[1]                          # 0
    w = zeros(n)                            # 0
    b = 0                                   # 0
    
    for i in 1:iterations                   # i*   [Loop]
        y_pred = zeros(m)                   # 0
        y_res = y - y_pred                  # m         [Vector Subtraction (m x 1) - (m x 1)]
        
        for k in 1:m                        # m*        [Loop]
            dp = dot(X[k,:], w)             # n            [(1 x n).(1 x n)]
            dp += b                         # 1            [Scalar Addition]
            y_pred[k] = dp                  # 0
        end
        
        # Calculate gradients
        dW = zeros(n)                       # 0
        for j in 1:n                        # n*        [Loop]
            dp_XY = dot(X[:,j], y_res)      # m             [Dot Product (m x 1).(m x 1)]
            dp_XY *= -2                     # 1
            
            if w[j] > 0                     # -             [IF/ELSE: Flops / Iteration = 2]            
                XY_pen = dp_XY + l1_penalty # 1                 [Scalar Addition]
                dW[j] = XY_pen / m          # 1                 [Scalar Division]
            else
                XY_pen = dp_XY - l1_penalty # 1                 [Scalar Subtraction]
                dW[j] = XY_pen / m          # 1                 [Scalar Subtraction]
            end
        end
        
        db = sum(y_res)                     # m         [Summation of (m x 1)]
        db *= -2                            # 1         [Scalar Multiplication]
        db /= m                             # 1         [Scalar Division]
        
        lrdW = learning_rate*dW             # 1         [Scalar Multiplication]
        w -= lrdW                           # 1         [Scalar Subtraction]
        
        lrdb = learning_rate*db             # 1         [Scalar Multiplication]
        b -= lrdb                           # 1         [Scalar Subtraction]
        
    end
    
    return w, b                             # O(1)   [Return Values]
    
end

# Flop Count:
# Variables: m: rows, n: columns, i: iterations
# i * (m + m*(n + 1) + n*(m + 1 + 1 + 1) + m + 1 + 1 + 1 + 1 + 1)

# Total Flops = i * (2mn + 3m + 3n + 6)



# Flop Count References:
# Flops For nxn Gaussian Elimination: http://web.mit.edu/18.06/www/Fall15/Matrices.pdf
