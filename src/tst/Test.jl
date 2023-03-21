module Test
export linear_regression_test, linear_regression_test_GD

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

#for performing linear regression on some data X with provided coefficients w
function linear_regression_test(X,w)
    
    pred = X * w
    
    return pred
    
end


#for performing linear regression on some data X with provided coefficients w for gradient descent regression
function linear_regression_test_GD(X,w,b)
    
    m = size(X)[1]
    pred = zeros(m)
    for k in 1:m
        pred[k] = dot(X[k,:], w) + b
    end
    
    return pred
    
end
