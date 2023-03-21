module TimeComplexity
export ridge_TC, lasso_TC

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

# Ridge Regression Function Decomposed to show computations:
# Parameters- X: (m x n), y: (m x 1), lambda: constant
function ridge_TC(X,y,lambda)
    s  = size(X)[2]                          # O(1)      [CHANGED [1] to [2]... Correct (grab m or n)?]
    Im = 1 * Matrix(I, s, s)                 # O(n^2)    [Constructing (n x n) I Matrix]
    xT = transpose(X)                        # O(1)      [(m x n)] => (n x m)
    xT_X = xT * X                            # O(m*n^2)  [(n x m) * (m x n)] => (n x n)
    l_Im = lambda * Im                       # O(n^2)    [Scalar multiplication of (n x n) matrix]
    xT_lambda = l_Im + xT_X                  # O(n^2)    [Scalar addition of (n x n) matrix]
    inverse = inv(xT_lambda)                 # O(n^3)    [Inverse (Gauss Elimination) with matrix of size (n x n)]
    w = inverse * xT                         # O(n^2*m)  [(n x n) * (n x m)] => (n x m)
    w *= y                                   # O(m*n)    [(n x m) * (m x 1)] => (n x 1)
    pred = X * w                             # O(m*n)    [(m x n) * (n x 1)] => (m x 1)
    err = y - pred                           # O(m)      [(m x 1) - (m x 1)] => (m x 1)
    
    return w, pred, err                      # O(1)
end

# Time Complexities Added
# O(n^3) + O(n^2*m) + O(m*n^2) + O(n^2) + O(n^2) + O(n^2) + O(m*n) + O(m*n) + O(m) + O(1) + O(1) + O(1)

# Final Time Complexity: 
# O(n^3) + O(n^2*m)

# Note: If m >> n (Data points >> Features), time complexity can be reduced to O(m^2*n)

# Variable Dimensions
# --------------------------------
# lambda = constant
# (n x n) = xT_X, inverse, Im, l_Im xT_lambda
# (n x m) = xT
# (n x 1) = w
# (m x n) = X
# (m x 1) = y, pred, err





# Parameters - X: (m x n), y: (m x 1), lambda: constant, learning_rate: constant, iterations: constant
# Variables  - m: rows, n: cols, i: iterations
# Note: This is the "expanded" version of our function, allowing each instruction to have its own line.
function lasso_TC(X, y, lambda)
    learning_rate = .025                    # O(1)   [Variable Assignment]
    iterations = 8000                       # O(1)   [Variable Assignment]
    l1_penalty = lambda                     # O(1)   [Variable Assignment]
    n = size(X)[2]                          # O(1)   [Accessing Size Variable]
    m = size(X)[1]                          # O(1)   [Accessing Size Variable]
    w = zeros(n)                            # O(n)   [Creation of vector /w size n]
    b = 0                                   # O(1)   [Variable Assignment]
    
    for i in 1:iterations                   # O(i)   [Loop]
        y_pred = zeros(m)                   # O(m)     [Creation of vector /w size m]
        y_res = y - y_pred                  # O(m)     [Vector Subtraction (m x 1) - (m x 1)]
        
        for k in 1:m                        # O(m)     [Loop]
            dp = dot(X[k,:], w)             # O(n)         [(1 x n).(1 x n)]
            dp += b                         # O(1)         [Scalar Addition]
            y_pred[k] = dp                  # O(1)         [Value Assignment]
        end
        
        # Calculate gradients
        dW = zeros(n)                       # O(n)      [shape of the params (feature #)]
        for j in 1:n                        # O(n)      [Loop]
            dp_XY = dot(X[:,j], y_res)      # O(m)          [Dot Product (m x 1).(m x 1)]
            dp_XY *= -2                     # O(1)          [Scalar Multiplication]
            
            if w[j] > 0                     # O(1)          [Accessing index] [IF/ELSE: Calculations / Iteration = 2*O(1)]
                XY_pen = dp_XY + l1_penalty # O(1)          [Scalar Addition]
                dW[j] = XY_pen / m          # O(1)          [Scalar Division]
            else
                XY_pen = dp_XY - l1_penalty # O(1)          [Scalar Subtraction]
                dW[j] = XY_pen / m          # O(1)          [Scalar Subtraction]
            end
        end
        
        db = sum(y_res)                     # O(m)      [Summation of (m x 1)]
        db *= -2                            # O(1)      [Scalar Multiplication]
        db /= m                             # O(1)      [Scalar Division]
        
        lrdW = learning_rate*dW             # O(1)      [Scalar Multiplication]
        w -= lrdW                           # O(1)      [Scalar Subtraction]
        
        lrdb = learning_rate*db             # O(1)      [Scalar Multiplication]
        b -= lrdb                           # O(1)      [Scalar Subtraction]
        
    end
    
    return w, b                             # O(1)   [Return Values]
    
end

# Variable Dimensions
# ---------------------
# Constant: learning_rate, iterations, l1_penalty, n, m, b, db, lrdW, lrdb, lambda
# (n x 1) : w, dW
# (m x 1) : y_pred, y_res, y
# (m x n) : X

# Total Time Complexity
# (6*O(1) + O(n)) + i*(O(m) + O(m) + m(O(n) + O(1) + O(1)) + O(n) + n(O(m) + O(1) + O(1) + O(1) + O(1)) + O(m) + 7*O(1))
# (6*O(1) + O(n)) + i*(5*O(m) + 2*O(m*n) + 5*O(n) + 7*O(1))
# (2*O(m*n*i) + 5*O(m*i) + 5*O(n*i) + 7*O(i)) + (6*O(1) + O(n))
# O(m*n*i) + O(m*i) + O(n*i) + O(i) + O(n) + O(1) 

# Final Time Complexity
# O(m*n*i)

# Time Complexity References
# ----------------------------------------
# Size(X)       : O(1)
# References    : [https://stackoverflow.com/questions/21614298/what-is-the-runtime-of-array-length, https://blog.finxter.com/python-list-length-whats-the-runtime-complexity-of-len/, ]

# Transpose(X)  : O(1)    
# References    : [https://www.mathworks.com/matlabcentral/answers/495668-what-s-the-transpose-complexity-big-o-in-matlab, https://stackoverflow.com/questions/61157101/in-julia-transpose-operator]

# Inverse(X)    : Worst Case-O(n^3) (Gauss Elimination), Best Case O(n^2.373)
# References    : [https://stackoverflow.com/questions/54890422/inv-versus-on-julia]

# Matrix *      : (m x n) * (n * p) => O(n*m*p), O(n^3)-O(n^2.72...)
# References    : [https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra]

# Matrix -      : O(m*n)
# References    : [https://www.geeksforgeeks.org/different-operation-matrices/]

# Matrix(I,s,s) : O(n^2)
# References    : [https://stackoverflow.com/questions/282926/time-complexity-of-memory-allocation]

# Zeros(n)      : O(n)
# References    : [https://discourse.julialang.org/t/faster-zeros-with-calloc/69860/13, https://stackoverflow.com/questions/5640850/java-whats-the-big-o-time-of-declaring-an-array-of-size-n]

# Dot(n, n)     : O(n)
# References    : [https://helloacm.com/teaching-kids-programming-compute-the-dot]
