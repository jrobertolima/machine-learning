function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% J(theta) = 1/m * SUM[-y*log(h(x) - (1 - y)*log(1 - h(x))]

h = sigmoid(X * theta); % Calculating h(x) using sigmoid function: logistic regression

py1 = y' * log(h); %' Calculating -y'log(h)  'part one of J

py0 = (1-y)' * log(1-h); %' Calculating -(1-y)'log(1-h) - 'part 2 of J 

J = 1/m * (-py1 - py0); %Finally, calculating J(theta)

grad = 1/m * X' * (h - y); %'Partial derivative of J

% =============================================================

end
