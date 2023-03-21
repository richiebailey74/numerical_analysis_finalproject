module Evaluation
export r_squared

using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics

function r_squared(targets, predictions)
    
    mean_target = mean(targets)
    
    ssr = 0
    sst = 0
    
    for i in 1:size(targets)[1]
        
        ssr += (targets[i] - predictions[i])^2
        sst += (targets[i] - mean_target)^2
        
    end
    
    r_sq = 1 - (ssr/sst)
    
end
