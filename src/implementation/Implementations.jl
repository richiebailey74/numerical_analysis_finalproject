module Implementations
export linear_regression_ridge, linear_regression_lasso_GD

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

function linear_regression_ridge(X,y,lambda)
    
    s = size(X)[2]
    Im =1* Matrix(I, s, s)
    
    w = inv(transpose(X)*X + lambda*Im) * transpose(X) * y
    
    pred = X * w
    
    err = y - pred
    
    return w    
    
end


function linear_regression_lasso_GD(X, y, lambda)
    
    learning_rate = .025
    iterations = 8000
    l1_penalty = lambda
    n = size(X)[2] # feature number
    m = size(X)[1] #sample number
    w = zeros(n) # shape of the params (feature #)
    b = 0
    
    for i in 1:iterations        
        y_pred = zeros(m)
        for k in 1:m
            y_pred[k] = dot(X[k,:], w) + b
        end
        
        #calculate gradients
        dW = zeros(n) # shape of the params (feature #)
        for j in 1:n
            if w[j] > 0  
                dW[j] = ( -1 * (2 * (dot(X[:,j], y - y_pred) ) ) + l1_penalty) ./ m
            else
                dW[j] = ( -1 * (2 * (dot(X[:,j], y - y_pred) ) ) - l1_penalty) ./ m
            end
        end
        
        db = - 2 * sum(y - y_pred) ./ m
        
        w = w - learning_rate*dW
        b = b - learning_rate*db
        
    end
    
    return w, b
    
end
