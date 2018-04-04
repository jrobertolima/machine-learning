function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% Calculating lambda/2mSUM(theta²) => reg_term

cost_reg_term = lambda/(2*m) * (sum(theta([2,size(theta,1)]).^2));

h = sigmoid(X * theta); % Calculating h(x) using sigmoid function: logistic regression

py1 = y' * log(h); %' Calculating -y'log(h)  'part one of J

py0 = (1-y)' * log(1-h); %' Calculating -(1-y)'log(1-h) - 'part 2 of J 

J = 1/m * (-py1 - py0) + cost_reg_term; %Finally, calculating J(theta)

%Calculating gardient regularization term
grad_reg_term = theta*(lambda/m);
grad_reg_term(1) = 0; % Zeroing theta(1)

grad = (1/m * X' * (h - y)) + grad_reg_term; %'Partial derivative of J

% =============================================================

end
