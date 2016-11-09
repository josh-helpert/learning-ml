%y = data(:, 2);
%X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
%theta = zeros(2, 1); % initialize fitting parameters
%alpha = 0.01;
%iterations = 1500;
%theta = gradientDescent(X, y, theta, alpha, iterations);

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%["h(theta):" num2str(rows(h)) "x" num2str(columns(h))] % 91x1
%["err:" num2str(rows(err)) "x" num2str(columns(err))]  % 91x1
%["theta:" num2str(rows(theta)) "x" num2str(columns(theta))]  % 2x1
%["X:" num2str(rows(X)) "x" num2str(columns(X))]        % 97x2
%["X':" num2str(rows(X')) "x" num2str(columns(X'))]     % 2x97

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.

    h = X * theta; % 97x1
    err = h .- y;  % 97x1
    % X' 2x97

    % TODO: Look over this and assure that this is using the sum over 'm'
    theta_delta = (alpha / m) * X' * err;
    %theta_delta
    theta = theta - theta_delta;
    %theta

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
