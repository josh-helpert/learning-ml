function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

%X:      5000x400
%Theta1: 25x401
%Theta2: 10x26

% Compute neural layers
bias = ones(m,1);
X    = [bias X];
a_2  = sigmoid(X * Theta1'); % 5000x25
a_2  = [bias a_2];
a_3  = sigmoid(a_2 * Theta2'); % 5000x10

% Compute which classifier (1,2..K=10) has the highest probability
[prob, p] = max(a_3, [], 2); % [prob=probability of index, p=index of max]

% =========================================================================

end
