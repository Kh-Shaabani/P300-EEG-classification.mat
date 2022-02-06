function [ Y_RC,Y_Char ] = mySTDA_test( X,W1_op,W2_op,w )
%Spatial-Temporal Discriminant Analysis Classifier for two classes of P300
%   X:matrix of test data,size(X)=[D1,D2,I,J,K]=[spatial feature,temporal feature,number of rows & columns,...
%   ...number of repetition of each row & column,number of character epochs].
%   W1_op:optimal W1(W in spatial space)(D1*L).
%   W2_op:optimal W2(W in temporal space)(D2*L).
%   w:projection vector(L^2*1).
%   Y_RC:row-column labeles of X, size(Y_RC)=[2,K].
%   Y_Char:character labels of X, size(Y_Char)=[1,K].

for k=1:size(X,5)
    for i=1:size(X,3)
        for j=1:size(X,4)
            A=(W1_op'*X(:,:,i,j,k))*W2_op;        % size(A)=[L,L]
            F(:,i,j,k)=A(:);      % size(F)=[L^2,I,J,K]
        end
    end
end
[ Y_RC,Y_Char ] = myLDA_testIII( F,w );

end

