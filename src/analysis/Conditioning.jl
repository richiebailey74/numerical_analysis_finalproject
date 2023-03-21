module Conditioning
export ridge_conditioning, lasso_gd_conditioning

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

# This Function checks the conditioning of the Ridge Regression
# by checking each individual operation to see if it is well conditioned.
# If each individual operation of the regression is well conditioned, 
# then the entire algorithm is well conditioned. If a part of the 
# regression is ill conditioned then the algorithm is not optimal.
function ridge_conditioning(X,y,lambda)
    k = zeros(0)                                           # vector containing all condition numbers

    s  = size(X)[2]
    push!(k,1)                                                       # s  = size(X)[2]
    
    Im = 1 * Matrix(I, s, s)
    push!(k,1)                                                       # Im = 1 * Matrix(I, s, s)
    
    w = inv(transpose(X)*X + lambda*Im) * transpose(X) * y
    A = transpose(X)*X
    push!(k, norm(transpose(X)*X) * norm(inv(transpose(X)*X)))       # xT * x
    b = lambda*Im
    push!(k, norm(lambda) * norm(inv(lambda)))                       # lambda * I
    C = A + b
    push!(k, norm(A)*norm(inv(A)) + norm(b)*norm(inv(b)))            # A + b
    D = inv(C)
    push!(k, norm(inv(C))*norm(C))                                   # inv(C)
    E = transpose(X) * y
    push!(k, (norm(transpose(X)) * norm(y)) / norm(transpose(X) * y))# xT * y
    F = D * E
    push!(k, (norm(D) * norm(E))/norm(D*E))                          # D * E
    
    pred = X * w
    push!(k, (norm(X) * norm(w))/norm(X*w))                           # X * w
        
    err = y - pred
    push!(k, norm(err))                                              # y - pred
    
    return k
end


function lasso_gd_conditioning(X, y, lambda)
    learning_rate = .025
    iterations = 8000
    l1_penalty = lambda
    n = size(X)[2] # feature number
    m = size(X)[1] #sample number
    w = zeros(n) # shape of the params (feature #)
    b = 0
    #Above declarations are all well-condtioned since they are assigning values to variables.
    cond = zeros(0); #This var will contain largest condition number from all operations

    for i in 1:iterations
        y_pred = zeros(m); #Well-conditioned since it's assigning zeros to variable
        for k in 1:m
            eq1 = dot(X[k,:], w);
            k1 = (norm(X[k,:]) * norm(w)) / dot(X[k,:], w); #Condition number for dot product is cond f(x) = 1/cos(x,y) = inverse cosine angle between both vectors
            push!(cond,k1)

            eq2 = eq1 + b;
            y_pred[k] = eq2;
            k2 = abs(eq1 / (eq2)); # abs(eq1 / (eq1 + b))
            push!(cond,k2)
        end

        #calculate gradients
        dW = zeros(n) # shape of the params (feature #)
        #Well-condtioned since it's assigning zeros to variable 
        for j in 1:n
            if w[j] > 0
                eq3 = y - y_pred; 
                #conditioning would be determined by the conditioning of of the subtraction of every respective element
                k3 = norm(y - y_pred);
                push!(cond,k3)

                eq4 = dot(X[:,j], eq3);
                k4 = (norm(X[:,j]) * norm(eq3)) / dot(X[:,j], eq3);
                push!(cond,k4)

                eq5 = 2 * eq4; #Condition Number 1 since multiplying by scalar.
                k5 = 1;
                push!(cond,k5)

                eq6 = -1 * eq5;
                #Condition Number 1 since multiplying by scalar.
                k6 = 1;
                push!(cond,k6)

                eq7 = eq6 + l1_penalty;
                k7 = norm(eq7);
                push!(cond,k7)

                eq8 = eq7 ./ m;
                dW[j] = eq8;
                #Condition number is 1 since it is the element division of two variables
                k8 = 1;
                push!(cond,k8)
            else
                eq9 = y - y_pred; 
                #conditioning would be determined by the conditioning of of the subtraction of every respective element
                k9 = norm(y - y_pred);
                push!(cond,k9)

                eq10 = dot(X[:,j], eq9);
                k10 = (norm(X[:,j]) * norm(eq9)) / dot(X[:,j], eq9);
                push!(cond,k10)

                eq11 = 2 * eq10; #Condition Number 1 since multiplying by scalar.
                k11 = 1;
                push!(cond,k11)

                eq12 = -1 * eq11;
                #Condition Number 1 since multiplying by scalar.
                k12 = 1;
                push!(cond,k12)

                eq13 = eq12 - l1_penalty;
                k13 = norm(eq13);
                push!(cond,k13)

                eq14 = eq13 ./ m;
                dW[j] = eq14;
                #Condition number is 1 since it is the element division of two variables
                k14 = 1;
                push!(cond,k14)
            end
                
            eq15 = y - y_pred;
            k15 = norm(y - y_pred);
            push!(cond,k15)

            eq16 = sum(eq15);
            k16 = 1;
            push!(cond,k16)

            eq17 = - 2 * eq16; #Scalar multiplicaiton in well-conditioned
            k17 = 1;
            push!(cond,k17)

            eq18 = eq17 ./ m; # dividing by a scalar m is well-conditioned
            db = eq18;
            k18 = 1;
            push!(cond,k18)

            eq19 = learning_rate*dW; #scalar multiplication is well-conditioned
            k19 = 1;
            push!(cond,k19)

            eq20 = w - eq19;
            w = eq20;
            k20 = norm(w - eq19);
            push!(cond,k20)

            eq21 = learning_rate*db; #scalar multiplication is well-conditioned
            k21 = 1; 
            push!(cond,k21)

            eq22 = b - eq21;
            b = eq22;
            k22 = norm(b - eq21);
            push!(cond,k22)

        end
        
    end
    return cond;
    
end
