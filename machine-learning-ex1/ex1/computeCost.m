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
%J(theta) = 1/2m*sum((h(theta)-y)^2)
%h(theta) = theta0 - theta1x

h = X * theta; % Matrix h(theta)
diff_h_y = h - y; %Difference h - y =>error matrix
diff_sqr = diff_h_y.^2; 
J = (1/(2*m))*sum(diff_sqr);



% =========================================================================

end
