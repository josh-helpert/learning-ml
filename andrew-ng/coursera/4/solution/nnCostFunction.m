function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1_vec_len = hidden_layer_size * (input_layer_size + 1);

Theta1 = reshape(nn_params(1:Theta1_vec_len), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + Theta1_vec_len):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Tranform y vector to Y matrix (size:# samples x # labels)
Y = zeros(m, num_labels);

for i = 1:m
  Y(i, y(i)) = 1;
end

% --- Forward propogation ---

%Layers 400 -> 25 -> 10
%X      5000 x 400
%y      5000 x 1
%Y      5000 x 10
%Theta1 25   x 401
%Theta2 10   x 26

bias = ones(m, 1);

a_1 = [bias X];            % 5000 x 401
z_2 = a_1 * Theta1';       % 5000 x 401 * 401 x 25 = 5000 x 25
a_2 = [bias sigmoid(z_2)]; % 5000 x 26
z_3 = a_2 * Theta2';       % 5000 x 26  *  26 x 10 = 5000 x 10
a_3 = sigmoid(z_3);        % 5000 x 10

% Compute regularization cost
Theta1_no_0 = Theta1(:, 2:end); % 25 x 400
Theta2_no_0 = Theta2(:, 2:end); % 10 x 25

J_reg_term = (lambda / (2 * m)) * (sum(sum(Theta1_no_0 .^ 2)) + sum(sum(Theta2_no_0 .^ 2)));

% Compute cost function
h = a_3; % Alias for clarity

J = (1 / m) * sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h))) + J_reg_term;

% --- Back-propogation ---

delta_3 = h - Y;                                           % 5000 x 10
delta_2 = (delta_3 * Theta2_no_0) .* sigmoidGradient(z_2); % 5000 x 10 * 10 x 25 = 5000 x 25

grad_1 = delta_2' * a_1; % 25 x 401
grad_2 = delta_3' * a_2; % 10 x 26

Theta1_grad = (1 / m) * (grad_1 + lambda * [zeros(size(Theta1, 1), 1) Theta1_no_0]);
Theta2_grad = (1 / m) * (grad_2 + lambda * [zeros(size(Theta2, 1), 1) Theta2_no_0]);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
