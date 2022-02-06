function [ W_FC,Y1_FC,Y2_FC,Y1,Y2 ] = myFC_train( X1,X2,L )
%Fisher’s Criterion-based spatial filtering training for two classes
%   Xi:trainig data of claas i, size(Xi)=[D1,D2,Ni]=[spatial feature,temporal feature,sample].
%   L:number of retained projection matrices from EVD.
%   W_FC:projection matrix of FC_based spatial filtering(D1*L).
%   Y1_FC:claas1 new data with decreased spatial feature dimention, size(Y1_FC)=[L,D2,N1]
%   Y2_FC:claas2 new data with decreased spatial feature dimention, size(Y1_FC)=[L,D2,N2]
%   Y1:claas1 new data with decreased spatial feature dimention, size(Y1)=[L*D2,N1]
%   Y2:claas2 new data with decreased spatial feature dimention, size(Y2)=[L*D2,N2]

X_c={X1,X2};
X=cat(3,X1,X2);     % size(X)=[D1,D2,N1+N2]
Mu1=mean(X1,3);     % size(Mu1)=[D1,D2]
Mu2=mean(X2,3);     % size(Mu2)=[D1,D2]
Mu_c(:,:,1)=Mu1;
Mu_c(:,:,2)=Mu2;
Mu=mean(X,3);       % size(Mu)=[D1,D2]
N1=size(X1,3);      % number of samples in class1
N2=size(X2,3);      % number of samples in class2
n=[N1,N2];

% beetween & within class scatter
 Sb=0;
 Sw_c=0;
 Sw=0;
 for c=1:2
    Sb_c=n(c)*(Mu_c(:,:,c)-Mu)*(Mu_c(:,:,c)-Mu)';
%     Sb_c=(Sb_c+Sb_c')/2;
    Sb=Sb+Sb_c;           % size(Sb)=[D1,D1]
    for j=1:n(c)
        Sw_j=(X_c{c}(:,:,j)-Mu_c(:,:,c))*(X_c{c}(:,:,j)-Mu_c(:,:,c))';
%         Sw_j=(Sw_j+Sw_j')/2;
        Sw_c=Sw_c+Sw_j;
    end
    Sw=Sw+Sw_c;      % size(Sw)=[D1,D1]
 end
 
% Eigendecomposition
[V,A]=eig(Sb,Sw);       % size(V)=[D1,D1]
[A,indx]=sort(diag(A),'descend');
V=V(:,indx);
W_FC=V(:,L);        % size(W_FC)=[D1,L]

for i=1:N1
    Y1_FC(:,:,i)=W_FC'*X1(:,:,i);     % size(Y1_FC)=[L,D2,N1]
end
for i=1:N2
    Y2_FC(:,:,i)=W_FC'*X2(:,:,i);     % size(Y2_FC)=[L,D2,N2]
end

Y1=[];
cnt=0;
for i=1:size(Y1_FC,2)
    for j=1:size(Y1_FC,1)
        cnt=cnt+1;
         Y1(cnt,:)=squeeze(Y1_FC(j,i,:));      % size(Y1)=[L*D2,N1]
    end
end

Y2=[];
cnt=0;
for i=1:size(Y2_FC,2)
    for j=1:size(Y2_FC,1)
        cnt=cnt+1;
        Y2(cnt,:)=squeeze(Y2_FC(j,i,:));      % size(Y2)=[L*D2,N2]
    end
end

end

