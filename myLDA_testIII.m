function [ Y_RC,Y_Char ] = myLDA_testIII( X,w )        % myLDA_testIII
%Linear Discriminant Analysis Classifier for two classes of P300
%   X:matrix of test data,size(X)=[D,I,J,K]=[feature,number of rows & columns,...
%   ...number of repetition of each row & column,number of character epochs].
%   w:projection vector(D*1).
%   Y_RC:row-column labeles of X, size(Y_RC)=[2,K].
%   Y_Char:character labels of X, size(Y_Char)=[1,K].

for k=1:size(X,4)
    for i=1:size(X,2)
        S(:,i,k)=w'*(squeeze(X(:,i,:,k)));    % classification score, size(S)=[J,I,K]
    end
end
S=squeeze(mean(S,1));           % size(S)=[I,K]
[S_c,column]=max(S(1:6,:));     % size(column)=[1,K]
[S_r,row]=max(S(7:12,:));       % size(row)=[1,K]
row=row+6;
Y_RC=[column;row];

for i=1:size(Y_RC,2)
    Y_Char(:,i)=(6*(Y_RC(2,i)-7))+Y_RC(1,i);
end

end


