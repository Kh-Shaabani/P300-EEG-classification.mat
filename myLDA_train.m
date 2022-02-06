function [ w,b ] = myLDA_train( X1,X2,T )
%Linear Discriminant Analysis training for two classes
%   Xi:matrix of training data of class i, size(Xi)=[D,Ni]=[feature,sample].
%   T:threshold.
%   w:projection vector(D*1).
%   b:bias term.

N1=size(X1,2);      % number of samples in class1
N2=size(X2,2);      % number of samples in class2
N=N1+N2;            % number of total samples

Mu1=mean(X1,2);       % empirical mean vector of class1, size(Mu1)=[D,1]
Mu2=mean(X2,2);       % empirical mean vector of class2, size(Mu2)=[D,1]

Sigma1=cov(X1');        % empirical covariance matrix of class1, size(Sigma1)=[D,D]
Sigma2=cov(X2');        % empirical covariance matrix of class2, size(Sigma2)=[D,D]
Sigma=(N1/N)*Sigma1+(N2/N)*Sigma2;        % common covariance matrix, size(Sigma)=[D,D]

w=inv(Sigma)*(Mu1-Mu2);     % size(w)=[D,1]
b=(1/2)*((T)-(Mu1'*inv(Sigma)*Mu1)+(Mu2'*inv(Sigma)*Mu2));      % a number

end

