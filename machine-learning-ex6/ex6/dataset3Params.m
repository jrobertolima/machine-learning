function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%Defining vector of values to try with sigma and C
values = [0.01 0.03 0.1 0.3 1 3 10 30]';%'
error_temp = 100;
values_size = size(values,1);

%Iterating over values to try every combination of sigma and C
for i = 1:values_size  
    for j = 1:values_size
        %Training the data set
        model= svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j)));

        %Predicting error values in cross validation set 
        predictions = svmPredict(model, Xval);
        errors = mean(double(predictions ~= yval));

        %Defining minimal values        
        if errors < error_temp
            C_temp = values(i);
            sigma_temp = values(j);
            error_temp = errors;
         end;    
    end;

end;
C = C_temp;
sigma = sigma_temp;
% =========================================================================
end
