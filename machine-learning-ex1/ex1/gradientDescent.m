function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

syms k
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum1 = 0;
    sum2 = 0;
    for k = 1:m
        sum1 = sum1 + (alpha/m)*X(k,1)*(theta'*X(k,:)'-y(k));
        sum2 = sum2 + (alpha/m)*X(k,2)*(theta'*X(k,:)'-y(k));
    end
    theta(1) = theta(1) - sum1;
    theta(2) = theta(2) - sum2;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
