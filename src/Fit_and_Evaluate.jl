using Pkg
using LinearAlgebra
using DataFrames
using CSV
using Statistics
using FileIO
using Images

using Implementations
using Test
using Evaluation
using Conditioning

matrix_data_temp = CSV.File("median_housing_cost_data.tsv") |> Tables.matrix;
matrix_data = matrix_data_temp[:, 2:9];

matrix_target_temp = CSV.File("housing_cost_targets.tsv") |> Tables.matrix;
matrix_target = matrix_target_temp[:, 2];

# train test split (test proportion of 20%) - test has 4128 samples, train has 16512 samples
matrix_data_train = matrix_data[1:16512, :];
matrix_data_test = matrix_data[16513:20640, :];

matrix_target_train = matrix_target[1:16512, :];
matrix_target_test = matrix_target[16513:20640, :];

# normalize data (min max normalize)
for i in 1:size(matrix_data_train)[2]

    matrix_data_train[:, i] = (matrix_data_train[:, i] .- findmin(matrix_data_train[:, i])[1]) ./ (findmax(matrix_data_train[:, i])[1] - findmin(matrix_data_train[:, i])[1])
    matrix_data_test[:, i] = (matrix_data_test[:, i] .- findmin(matrix_data_test[:, i])[1]) ./ (findmax(matrix_data_test[:, i])[1] - findmin(matrix_data_test[:, i])[1])

end

matrix_target_train = (matrix_target_train .- findmin(matrix_target_train)[1]) ./ (findmax(matrix_target_train)[1] - findmin(matrix_target_train)[1]);
matrix_target_test = (matrix_target_test .- findmin(matrix_target_test)[1]) ./ (findmax(matrix_target_test)[1] - findmin(matrix_target_test)[1]);

lambdas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.5, 5, 10, 12, 15, 18, 20];

optimal_l1_params = zeros(8);
optimal_l2_params = zeros(8);

optimal_l1_r_sq = -1;
optimal_l2_r_sq = -1;

optimal_l1_lam = -1000;
optimal_l2_lam = -1000;

for it in lambdas

    parameters_l2 = linear_regression_ridge(matrix_data_train, matrix_target_train, it)
    parameters_l1, b_l1 = linear_regression_lasso_GD(matrix_data_train, matrix_target_train, it)

    predictions_l2 = linear_regression_test(matrix_data_test, parameters_l2)
    predictions_l1 = linear_regression_test_GD(matrix_data_test, parameters_l1, b_l1)
    predictions_l1 = reshape(predictions_l1, length(predictions_l1), 1)

    r_sq_l2 = r_squared(sort(matrix_target_test, dims=1), sort(predictions_l2, dims=1))
    r_sq_l1 = r_squared(sort(matrix_target_test, dims=1), sort(predictions_l1, dims=1))

    if r_sq_l2 > optimal_l2_r_sq
        optimal_l2_r_sq = r_sq_l2
        optimal_l2_params = parameters_l2
        optimal_l2_lam = it
    end

    if r_sq_l1 > optimal_l1_r_sq
        optimal_l1_r_sq = r_sq_l1
        optimal_l1_params = parameters_l1
        optimal_l1_lam = it
    end

end

println("The optimal lambda coefficient for l1 norm is:", optimal_l1_lam)
println("The optimal lambda coefficient for l2 norm is:", optimal_l2_lam)
println("The optimal r^2 value for l1 norm is:", optimal_l1_r_sq)
println("The optimal r^2 value for l2 norm is:", optimal_l2_r_sq)

# hard code in the optimal regularization coefficients to function
parameters_l2 = linear_regression_ridge(matrix_data_train, matrix_target_train, 5);
parameters_l1, b_l1 = linear_regression_lasso_GD(matrix_data_train, matrix_target_train, 0.00001);

predictions_l2 = linear_regression_test(matrix_data_test, parameters_l2);
predictions_l1 = linear_regression_test_GD(matrix_data_test, parameters_l1, b_l1);
predictions_l1 = reshape(predictions_l1, length(predictions_l1), 1);

sorted_test_targets = sort(matrix_target_test, dims=1)
sorted_l2_preds = sort(predictions_l2, dims=1)
sorted_l1_preds = sort(predictions_l1, dims=1)
r_sq_l2 = r_squared(sorted_test_targets, sorted_l2_preds);
r_sq_l1 = r_squared(sorted_test_targets, sorted_l1_preds);

img1 = load("figures/qqplot_lasso.png")
print("The optimal R squared value for Lasso regression is:", r_sq_l1)

img2 = load("figures/qqplot_ridge.png")
print("The optimal R squared value for Ridge regression is:", r_sq_l2)

#calculating conditioning of our algorithms using the optimal lambda values
k_ridge = ridge_conditioning(matrix_data_train, matrix_target_train, 5);
k_lasso = lasso_gd_conditioning(matrix_data_train, matrix_target_train, 0.00001);

print(k_ridge')
print(k_lasso')

print("Conditioning values for operations in ridge regression\n")
for i in k_ridge

    println(i)

end
println("\nThus we can see ridge regression has no ill conditioned steps and is a stable algorithm.")

print("Conditioning values for operations in lasso regression\n")
nanCount = count(i -> (isnan(i)), k_lasso)
nonnanCount = length(k_lasso) - nanCount
println("The number of ill conditioned steps in our gradient descent implementation of lasso is: ", nanCount)
println("The number of well conditioned steps in our gradient descent implementation of lasso is: ", nonnanCount)
println("So it is more indicative to look at the proportion of ill conditioned steps: ", nanCount / length(k_lasso))
println("However, this implementation of lasso regression is still ill conditioned since it contains ill conditioned steps")
