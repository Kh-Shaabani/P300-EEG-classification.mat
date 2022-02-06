function [ Y_FC,Y ] = myFC_test( X,W_FC )
%Fisher’s Criterion-based spatial filtering for test deta of two classes of P300
%   X:matrix of test data,size(X)=[D1,D2,I,J,K]=[spatial feature,temporal feature,number of rows & columns,...
%   ...number of repetition of each row & column,number of character epochs].
%   w_FC:projection vector of FC_based spatial filtering(D1*L).
%   Y_FC:output of , size(Y_FC)=[L,D2,I,J,K].
%   Y:size(Y)=[L*D2,I,J,K].

for k=1:size(X,5)
    for i=1:size(X,3)
        for j=1:size(X,4)
            Y_FC(:,:,i,j,k)=W_FC'*X(:,:,i,j,k);      % size(Y_FC)=[L,D2,I,J,K]
        end
    end
end

Y=[];
cnt=0;
for i=1:size(Y_FC,2)
    for j=1:size(Y_FC,1)
        cnt=cnt+1;
        Y(cnt,:,:,:)=squeeze(Y_FC(j,i,:,:,:));      % size(Y)=[L*28,12,15,45]
    end
end

end

