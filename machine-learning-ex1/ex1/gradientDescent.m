function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %θ1−α1m∑i=1m((hθ(xi)−yi)xi)
%implementation using vectorization
    h = X * theta;
    diff_mul_x = X'* (h - y); %'
    theta = theta - (alpha*1/m)*diff_mul_x;
%end of implementation

%implementation using loop
%    for i_theta = 1:2
%        diff_mul_x = 0;
%        for i = 1:m
%            diff_mul_x += ((theta(1) + theta(2)*X(i,2)) - y(i))*X(i,i_theta);
%        end;
%        theta(i_theta) = theta(i_theta) - (alpha*1/m)*diff_mul_x;
%    end;  
%end of implementation

% ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
