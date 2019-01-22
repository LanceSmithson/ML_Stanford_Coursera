function [J grad] = nnCostFunction(nn_params, ...
                                   a_1_size, ...
                                   a_2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, a_2_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:a_2_size * (a_1_size + 1)), ...
                 a_2_size, (a_1_size + 1));

Theta2 = reshape(nn_params((1 + (a_2_size * (a_1_size + 1))):end), ...
                 num_labels, (a_2_size + 1));

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
%

Y = zeros(m,num_labels);
for i = 1:m
  Y(i,y(i)) = 1;
endfor;

a_1 = [ones(m, 1) X];
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2*Theta2';
prediction = sigmoid(z_3);

true = -Y.*log(prediction);
false = (1-Y).*log(1-prediction);
Ksum = sum(true-false,2);
non_pen_J = sum(Ksum)/m;

penalty_theta_1 = [Theta1(:,2:end)];
penalty_1_sum = sum(sum(penalty_theta_1.^2,2));

penalty_theta_2 = [Theta2(:,2:end)];
penalty_sum_2 = sum(sum(penalty_theta_2.^2,2));

penalty = (lambda / (2*m))*(penalty_1_sum + penalty_sum_2);

J = non_pen_J + penalty;

sig_3 = prediction.-Y;
sig_2 = (sig_3*Theta2).*sigmoidGradient([ones(size(z_2, 1), 1) z_2]);
sig_2 = sig_2(:, 2:end);

delta_1 = (sig_2'*a_1);
delta_2 = (sig_3'*a_2);

penalty_1 = (lambda/m)*[Theta1_grad(1:end,1), Theta1(:, 2:end)];
penalty_2 = (lambda/m)*[Theta2_grad(1:end,1), Theta2(:, 2:end)];
Theta1_grad = delta_1./m + penalty_1;
Theta2_grad = delta_2./m + penalty_2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end