function [ W_CSP,Y1_CSP,Y2_CSP,Y1,Y2 ] = myCSP_train( X1_CSP,X2_CSP,m )
%Common Spatial Pattern spatial filtering training for two classes
%   Xi_CSP:trainig data of claas i, size(Xi_CSP)=[D1,D2,Ni]=[spatial feature,temporal feature,sample].
%   m:number of spatial filters.
%   W_CSP:projection matrix of FC_based spatial filtering(D1*L).
%   Y1_CSP:claas1 new data with decreased spatial feature dimention, size(Y1_FC)=[L,D2,N1].
%   Y2_CSP:claas2 new data with decreased spatial feature dimention, size(Y1_FC)=[L,D2,N2].
%   X1_LDA:claas1 new data with decreased spatial feature dimention, size(Y1)=[L*D2,N1].
%   X2_LDA:claas2 new data with decreased spatial feature dimention, size(Y2)=[L*D2,N2].

Cov1=0;
Cov2=0;
for i=1:size(X1_CSP,3)
    X1=X1_CSP(:,:,i);       % size(X1)=[D1,D2]
    X2=X2_CSP(:,:,i);       % size(X2)=[D1,D2]
    Mu1=mean(X1,2);         % size(Mu1)=[D1,1]
    Mu2=mean(X2,2);         % size(Mu2)=[D1,1]
    X1=X1-Mu1;              % size(X1)=[D1,D2]
    X2=X2-Mu2;              % size(X2)=[D1,D2]
    C1=cov(X1');            % size(C1)=[D1,D1]
    C2=cov(X2');            % size(C2)=[D1,D1]
    Cov1=Cov1+C1;           % size(Cov1)=[D1,D1]
    Cov2=Cov2+C2;           % size(Cov2)=[D1,D1]
end
Cov1=Cov1/size(X1_CSP,2);     % size(Cov1)=[D1,D1]
Cov2=Cov2/size(X2_CSP,2);     % size(Cov2)=[D1,D1]
Cov_c=Cov1+Cov2;                    % size(Cov_c)=[D1,D1]
[Uc,Lc]=eig(Cov_c);                 % size(Uc)=[D1,D1], % size(Lc)=[D1,D1]
G=Uc*((Lc)^(-0.5))';                % size(G)=[D1,D1]
S1=G'*Cov1*G;                       % size(S1)=[D1,D1]
[U,L1]=eig(S1);                     % size(U)=[D1,D1], % size(L1)=[D1,D1]
[L1,indx]=sort(diag(L1),'descend');     % size(L1)=[D1,1]
L2=ones(length(L1),1)-L1;               % size(L2)=[D1,1]
U=U(:,indx);                            % size(U)=[D1,D1]
W=G*U;                                  % size(W)=[D1,D1]
W_CSP=W(:,[1:m,end-m+1:end]);           % size(W_CSP)=[D1,2m]

for i=1:size(X1_CSP,3)
    Y1_CSP(:,:,i)=W_CSP'*X1_CSP(:,:,i);     % size(Y1_CSP)=[2m,D2,N1]
end
for i=1:size(X2_CSP,3)
    Y2_CSP(:,:,i)=W_CSP'*X2_CSP(:,:,i);     % size(Y2_CSP)=[2m,D2,N2]
end

Y1=[];
cnt=0;
for i=1:size(Y1_CSP,2)
    for j=1:size(Y1_CSP,1)
        cnt=cnt+1;
        Y1(cnt,:)=squeeze(Y1_CSP(j,i,:));      % size(Y1)=[2m*D2,N1]
    end
end

Y2=[];
cnt=0;
for i=1:size(Y2_CSP,2)
    for j=1:size(Y2_CSP,1)
        cnt=cnt+1;
        Y2(cnt,:)=squeeze(Y2_CSP(j,i,:));      % size(Y2)=[2m*D2,N2]
    end
end

end

