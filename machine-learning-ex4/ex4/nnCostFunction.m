function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOsuSTFUNCTION Implements the neural network cost function for a two layer
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
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
%

X = [ones(size(X,1),1), X];

ident = eye(num_labels);
ymatrix = ident(y,:);

z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2];
z3 = a2*Theta2';
h = sigmoid(z3);

for i = 1:m
    J = J + sum(-ymatrix(i,:).*log(h(i,:))-(ones(1,num_labels)-ymatrix(i,:)).*log(ones(1,num_labels)-h(i,:)));
end
J = (1/m)*J;

d3 = h-ymatrix;
d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);
delt1 = d2'*X;
delt2 = d3'*a2;
Theta1_grad = delt1/m;
Theta2_grad = delt2/m;

% regularization

regTheta1 = Theta1;
regTheta2 = Theta2;
regTheta1(:,1) = [];
regTheta2(:,1) = [];
regterm1 = 0;
regterm2 = 0;

for j = 1:hidden_layer_size
    for k = 1:input_layer_size
        regterm1 = regterm1 + regTheta1(j,k)^2;
    end
end

for j = 1:num_labels
    for k = 1:hidden_layer_size
        regterm2 = regterm2 + regTheta2(j,k)^2;
    end
end

J = J + lambda/(2*m)*(regterm1+regterm2);

reggrad1 = [zeros(size(Theta1(:,1))), (lambda/m)*regTheta1];
reggrad2 = [zeros(size(Theta2(:,1))), (lambda/m)*regTheta2];

Theta1_grad = Theta1_grad + reggrad1;
Theta2_grad = Theta2_grad + reggrad2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
