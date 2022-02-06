function [ Y_CSP,Y ] = myCSP_test( test_CSP,W_CSP )
%Common Spatial Pattern spatial filtering for test deta of two classes of P300
%   X:matrix of test data,size(X)=[D1,D2,I,J,K]=[spatial feature,temporal feature,number of rows & columns,...
%   ...number of repetition of each row & column,number of character epochs].
%   w_FC:projection vector of FC_based spatial filtering(D1*L).
%   Y_FC:output of , size(Y_FC)=[L,D2,I,J,K].
%   Y:size(Y)=[L*D2,I,J,K].

for k=1:size(test_CSP,5)
    for i=1:size(test_CSP,3)
        for j=1:size(test_CSP,4)
            Y_CSP(:,:,i,j,k)=W_CSP'*test_CSP(:,:,i,j,k);        % size(Y_CSP)=[2,28,12,15,45]
        end
    end
end

Y=[];
cnt=0;
for i=1:size(Y_CSP,2)
    for j=1:size(Y_CSP,1)
        cnt=cnt+1;
        Y(cnt,:,:,:)=squeeze(Y_CSP(j,i,:,:,:));      % size(Y)=[56,12,15,45]
    end
end

end

