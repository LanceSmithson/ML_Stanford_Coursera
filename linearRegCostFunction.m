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
%

prediction = X*theta;
error = sum((prediction - y).^2);

penalty = sum(theta(2:end).^2);

J = (error + (lambda)*penalty) / (2 * m);

grad = sum((prediction-y).*X);
grad_penalty = lambda * [0;theta(2:end)];
grad = (grad' + grad_penalty) / m;












% =========================================================================

grad = grad(:);

end
