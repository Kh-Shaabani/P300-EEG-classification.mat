function [ Y_RC,Y_Char ] = myLDA_testI( X,w )       % myLDA_testI=myLDA_testII
%Linear Discriminant Analysis Classifier for two classes of P300
%   X:matrix of test data,size(X)=[D,I,K]=[feature,number of rows & columns,...
%   ...,number of character epochs].
%   w:projection vector(D*1).
%   Y_RC:row-column labeles of X, size(Y_RC)=[2,K].
%   Y_Char:character labels of X, size(Y_Char)=[1,K].

for k=1:size(X,3)
    S(:,k)=w'*(squeeze(X(:,:,k)));    % classification score, size(S)=[I,K]
end
[S_c,column]=max(S(1:6,:));     % size(column)=[1,K]
[S_r,row]=max(S(7:12,:));       % size(row)=[1,K]
row=row+6;
Y_RC=[column;row];

for i=1:size(Y_RC,2)        % size(Y_RC,2)=K
    Y_Char(:,i)=(6*(Y_RC(2,i)-7))+Y_RC(1,i);
end

end

