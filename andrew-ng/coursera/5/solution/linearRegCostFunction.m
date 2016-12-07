function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

%X      12 x 2
%y      12 x 1
%theta  2  x 1
%lambda 1  x 1

% Hyp err term
h = X * theta;
err = h - y;

% J regularization term
theta_no_0 = [zeros(size(theta, 2)); theta(2:end, :)];

J_reg = lambda * theta_no_0' * theta_no_0;

% Cost
J = (1 / (2 * m)) * (err' * err + J_reg);

% Gradient
grad = (1 / m) * (X' * err + lambda * theta_no_0);

% =========================================================================

grad = grad(:);

end
