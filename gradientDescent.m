function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    prediction = sum((theta'.*X),2);
    difference = prediction - y;
    %difference_x = difference.*X(:,2);
    %summation = sum(difference);
    %summation_x = sum(difference_x);
    %dirv_of_cost_0 = summation/m;
    %dirv_of_cost_1 = summation_x/m;
    temp_0 = theta(1) - (alpha*sum(difference)/m);
    temp_1 = theta(2) - (alpha*sum(difference.*X(:,2))/m);
    theta = [temp_0;temp_1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

plot(J_history)

end
