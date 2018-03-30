function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.' 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% x = (x - mu)/sigma

n_col = size(X,2); %Features x1, x2... xn
n_lin = size(X,1); %Values of x..

for col = 1:n_col
    mu(1,col) = mean(X(:,col)); %Get the mean of column COL
    sigma(1,col) = std(X(:,col)); %Get the standard deviation of column COL

    for lin = 1:n_lin
        X_norm(lin, col) = (X(lin,col) - mu(1,col))/sigma(1,col); 
    end
% ============================================================
end
