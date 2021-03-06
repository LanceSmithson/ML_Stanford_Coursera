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


z = sum((theta' .* X),2);
prediction = sigmoid(z);
true = (-1 .* y) .* log(prediction);
false = (1 .- y) .* log(1 .- prediction);
cost = sum(true .- false) / m;

theta_penalty = [0 ; theta(2:size(theta), :)];
theta_squared = theta_penalty.^2;
penalty = sum(theta_squared) * (lambda / (2 * m));

J = cost + penalty;

difference = prediction .- y;
grad_sum = sum(difference .* X);
grad_penalty = lambda * theta_penalty;
grad = (grad_sum' + grad_penalty) / m;

% =============================================================

end


