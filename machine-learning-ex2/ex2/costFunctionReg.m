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

syms k x
h = @(x) theta'*x';
sumj = 0;
sumtheta = 0;

for i = 2:size(theta)
    sumtheta = sumtheta + theta(i)^2;
end

for k = 1:m
    sumj = sumj + (-y(k)*log(sigmoid(h(X(k,:))))-(1-y(k))*log(1-sigmoid(h(X(k,:)))));
end
J = (1/m)*sumj + sumtheta*lambda/(2*m);

for i=1:size(theta)
    sumg = 0;
    for k = 1:m
        sumg = sumg + (sigmoid(h(X(k,:)))-y(k))*X(k,i);
    end
    grad(i)=sumg/m;
end

for i=2:size(theta)
    grad(i)=grad(i)+theta(i)*lambda/m;
end


% =============================================================

end
