function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%J = (1/(2*m))*sum((h(x)-y)^2) \n

%h(x) = theta0 + theta1x
h = X * theta; 

%Difference between h(theta) and y (actual values): squared errors
diff_h_y_sqr = (h - y).^2; 

%Finally J
J = (1/(2*m))*(sum(diff_h_y_sqr));

% =========================================================================

end
