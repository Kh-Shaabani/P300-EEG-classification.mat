function [ w,b ] = mySKLDA_train( X1,X2,T )
%Shrinkage LDA training for two classes
%   Xi:matrix of training data of class i, size(X)=[D,N]=[feature,sample].
%   T:threshold.
%   w:projection vector(D*1).
%   b:bias term.

N1=size(X1,2);      % number of samples in class1
N2=size(X2,2);      % number of samples in class2
N=N1+N2;            % number of total samples
D=size(X1,1);       % dimensionality of the feature space

Mu1=mean(X1,2);       % empirical mean vector of class1, size(Mu1)=[D,1]
Mu2=mean(X2,2);       % empirical mean vector of class2, size(Mu2)=[D,1]

Sigma1=cov(X1');        % empirical covariance matrix of class1, size(Sigma1)=[D,D]
Sigma2=cov(X2');        % empirical covariance matrix of class2, size(Sigma2)=[D,D]

v1=(trace(Sigma1)/D);       % average eigenvalue of Sigma1, a number
v2=(trace(Sigma2)/D);       % average eigenvalue of Sigma2, a number

A1=0; A2=0; C1=0; C2=0;
for n=1:N1
    for i=1:D
        if n==1
            A1=A1+((Sigma1(i,i)-v1)^2);     % a number
        end
        for j=1:D
            z1(n)=(X1(i,n)-Mu1(i))*(X1(j,n)-Mu1(j));        % size(z1)=[N1,1]
            if n==1 && i~=j
                C1=C1+(Sigma1(i,j)^2);      % a number
            end
        end
    end
end
B1=var(z1);     % a number
landa1=(N1/(N1-1)^2)*(B1/(C1+A1));      % optimal shrinkage parameter for class1, a number
Sigma1_SK=(1-landa1)*Sigma1+landa1*v1*eye(D);       % Shrinkage covariance matrix of class1, size(Sigma1_SK)=[D,D]

for n=1:N2
    for i=1:D
        if n==1
            A2=A2+((Sigma2(i,i)-v2)^2);     % a number
        end
        for j=1:D
            z2(n)=(X2(i,n)-Mu2(i))*(X2(j,n)-Mu2(j));        % size(z2)=[N2,1]
            if n==1 && i~=j
                C2=C2+(Sigma2(i,j)^2);      % a number
            end
        end
    end
end

B2=var(z2);     % a number
landa2=(N2/(N2-1)^2)*(B2/(C2+A2));      % optimal shrinkage parameter for class2, a number
Sigma2_SK=(1-landa2)*Sigma2+landa2*v2*eye(D);       % Shrinkage covariance matrix of class2, size(Sigma2_SK)=[D,D]

Sigma_SK=(N1/N)*Sigma1_SK+(N2/N)*Sigma2_SK;         % common covariance matrix, size(Sigma_SK)=[D,D]

w=inv(Sigma_SK)*(Mu1-Mu2);      % size(w)=[D,1]
b=(1/2)*((T)-(Mu1'*inv(Sigma_SK)*Mu1)+(Mu2'*inv(Sigma_SK)*Mu2));        % a number

end

