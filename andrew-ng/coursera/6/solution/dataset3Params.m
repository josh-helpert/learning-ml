function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

if (C != 1 && sigma != 0.1)
  steps = [0.01 0.03 0.1 0.3 1 3 10 30]';
  num_steps = length(steps);
  min_i = 0;
  min_j = 0;
  min_err = -1;

  for i = 1:num_steps
    C = steps(i);

    for j = 1:num_steps
      ["Test i,j=(" num2str(i) ","  num2str(j) ")"]
      sigma = steps(j);

      model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
      predictions = svmPredict(model, Xval);
      err = mean(double(predictions != yval));

      if (min_i == 0 || err < min_err)
        ["new min (" num2str(err) ") found at: " num2str(i) ":" num2str(j)]

        min_i = i;
        min_j = j;
        min_err = err;
      endif
    end
  end

  C     = steps(min_i);
  sigma = steps(min_j);

  ["min C: "     num2str(C)     " at " num2str(min_i)]
  ["min sigma: " num2str(sigma) " at " num2str(min_j)]
endif

% =========================================================================

end
