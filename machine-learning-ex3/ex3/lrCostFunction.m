function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h=sigmoid(X*theta);

%J=1/m*[sum(-y.*log(h)-(1-y).*log(1-h))+1/2*lambda*sum(theta(2:size(theta)).^2);

J=1/m*[sum(-y.*log(h)-(1-y).*log(1-h))+lambda/2*sum(theta(2:size(theta)).^2)] ;	
grad=1/m*(X'*(h-y));
temp = theta; 
temp(1) = 0;
grad = grad + lambda/m*temp;