function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
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
Y=[ones(size(X,1),1) X];
z2=Theta1*Y';
a2=sigmoid(z2);
a2=a2';
a2=[ones(size(a2,1),1) a2];
z3=Theta2*a2';
a3=sigmoid(z3);
h=a3';
for i=1:num_labels,
	k=y==i;
	J=J+1/m*sum(-k.*log(h(:,i))-(1-k).*log(1-h(:,i)));
	end;	








initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
D3=zeros(size(initial_Theta2,1),size(initial_Theta2,2));
D2=zeros(size(initial_Theta1,1),size(initial_Theta1,2));
for t=1:m,
	A1=X'(:,t);
	A1=[1;A1];
	Z2=Theta1*A1;
	A2=sigmoid(Z2);
	A2=[1;A2];
	Z3=Theta2*A2;
	A3=sigmoid(Z3);
	ty=zeros(num_labels,1);
	ty(y(t))=1;
	delta3=A3-ty;
	Z2=[1;Z2];
	delta2=Theta2'*delta3.*sigmoidGradient(Z2);
	delta2=delta2(2:end);
	D3=D3+delta3*A2';
	D2=D2+delta2*A1';
end;

Theta1_grad=1/m*D2;
Theta2_grad=1/m*D3;





t1=Theta1;
t2=Theta2;
t1(:,1)=0;
t2(:,1)=0;
reg=sum(sum(t1.^2))+sum(sum(t2.^2));
J=J+reg*lambda/(2*m);

Theta1_grad=Theta1_grad+lambda/m*(t1);
Theta2_grad=Theta2_grad+lambda/m*(t2);










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
