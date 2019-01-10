function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    prediction = sum((theta'.*X),2);
    difference = prediction - y;
    
    %difference_x0 = difference.*X(:,1);
    %difference_x1 = difference.*X(:,2);
    %difference_x2 = difference.*X(:,3);
    difference_matrix = X.*difference;
    
    %summation_x0 = sum(difference_x0);
    %summation_x1 = sum(difference_x1);
    %summation_x2 = sum(difference_x2);
    summation_matrix = sum(difference_matrix);
    
    %dirv_of_cost_0 = summation_x0/m;
    %dirv_of_cost_1 = summation_x1/m;
    %dirv_of_cost_2 = summation_x2/m;
    dirv_of_cost_matrix = summation_matrix/m;
    
    %temp_0 = theta(1) - (alpha*sum(difference_x0)/m);
    %temp_1 = theta(2) - (alpha*sum(difference_x1)/m);
    %temp_2 = theta(3) - (alpha*sum(difference_x2)/m);
    temp_matrix = theta' - (alpha*dirv_of_cost_matrix);
    
    theta = temp_matrix';
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
