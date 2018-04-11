function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); %return

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%Calculating Cost regularization term
cost_reg_term = lambda/(2*m) * (sum(theta([2,size(theta,1)]).^2));

% Calculating h(x) using sigmoid function: logistic regression
h = sigmoid(X * theta);


% Calculating -y'log(h) part one of J(theta)  '
py1 = y' * log(h);

%' Calculating -(1-y)'log(1-h) - 'part 2 of J 
py0 = (1-y)' * log(1-h); %'

%Finally, calculating J(theta)
J = 1/m * (-py1 - py0) + cost_reg_term; 

%Calculating gardient regularization term
grad_reg_term = theta*(lambda/m);
grad_reg_term(1) = 0; % Zeroing theta(1)

grad = (1/m * X' * (h - y)) + grad_reg_term; %'Partial derivative of J

% =============================================================

grad = grad(:);

end
