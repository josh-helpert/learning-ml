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

%["Dim X:" num2str(rows(X)) "x" num2str(columns(X))]             % Log X dimensions
%["Dim theta:" num2str(rows(theta)) "x" num2str(columns(theta))] % Log theta dimensions

h = X * theta;
err = h .- y;
err_sq = err .^ 2;

J = (1 / (2 * m)) * sum(err_sq);

% =========================================================================

end
