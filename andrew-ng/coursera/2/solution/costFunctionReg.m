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

%X     m   x n+1
%X'    n+1 x m
%h     m   x 1
%y     m   x 1
%theta n+1 x 1
%grad  n+1 x 1

% Compute regularization cost
theta_rest = theta(2:end, :); % Exclude first from regularization
J_reg_term = (lambda / (2 * m)) * theta_rest' * theta_rest;

% Computer regularization for gradient
grad_rest = [0; ones(length(theta) - 1, 1)]; % One-liner to ignore regularization of first term
grad_reg_term = (lambda / m) * grad_rest .* theta;

% Compute
h       = sigmoid(X * theta);
J       = (1 / m) * sum(-y' * log(h) - (1 - y') * log(1 - h)) + J_reg_term;
grad    = (1 / m) * X' * (h - y) + grad_reg_term;

% =============================================================

end
