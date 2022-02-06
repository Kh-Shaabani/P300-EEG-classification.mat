function [ w,b,W1_op,W2_op,iteration ] = mySTDA_train( X1,X2,L,err )
%Spatial-Temporal Discriminant Analysis training for two classes
%   Xi:trainig data of claas i, size(Xi)=[D1,D2,Ni]=[spatial feature,temporal feature,sample].
%   L:number of retained projection matrices from EVD.
%   err:stop criterion.
%   w:projection vector(L^2*1).
%   b:bias term.
%   W1_op:optimal W1(W in spatial space)(D1*L).
%   W2_op:optimal W2(W in temporal space)(D2*L).
%   iteration:the iteration of the while loop to achieve convergence.

N1=size(X1,3);      % number of samples in class 1
N2=size(X2,3);      % number of samples in class 2
N=N1+N2;            % number of total samples
n=[N1,N2];
X=cat(3,X1,X2);
W2(:,:,1)=eye(size(X1,2));         % initialization for projection matrix in temporal dimention of features
% W1(:,:,1)=zeros(size(X1,1),L);
iteration=1;

while 1
  iteration=iteration+1;  
    for k=1:2
        if k==1
            for i=1:N
                Y_1(:,:,i)=X(:,:,i)*W2(:,:,iteration-1);         % k=1, size(Y_1)=[D1,L,N]
            end
            for i=1:N1
                Y_11(:,:,i)=X1(:,:,i)*W2(:,:,iteration-1);       % k=1, class=1, size(Y_11)=[D1,L,N1]
            end
            for i=1:N2
                Y_12(:,:,i)=X2(:,:,i)*W2(:,:,iteration-1);       % k=1, class=2, size(Y_12)=[D1,L,N2]
            end
            Y_1c={Y_11,Y_12};
            Mu_Y_1=mean(Y_1,3);         % size(Mu_Y_1)=[D1,L]
            Mu_Y_11=mean(Y_11,3);       % size(Mu_Y_11)=[D1,L]
            Mu_Y_12=mean(Y_12,3);       % size(Mu_Y_12)=[D1,L]
            Mu_Y_1c(:,:,1)=Mu_Y_11;
            Mu_Y_1c(:,:,2)=Mu_Y_12;
            
            % beetween & within class scatter
            Sb1=0;
            Sw_j=0;
            Sw1=0;
            for c=1:2
                Sb=n(c)*(Mu_Y_1c(:,:,c)-Mu_Y_1)*(Mu_Y_1c(:,:,c)-Mu_Y_1)';
                Sb=(Sb+Sb')/2;
                Sb1=Sb1+Sb;           % size(Sb1)=[D1,D1]
                for j=1:n(c)
                    Sw=(Y_1c{c}(:,:,j)-Mu_Y_1c(:,:,c))*(Y_1c{c}(:,:,j)-Mu_Y_1c(:,:,c))';
                    Sw=(Sw+Sw')/2;
                    Sw_j=Sw_j+Sw;
                end
                Sw1=Sw1+Sw_j;      % size(Sw1)=[D1,D1]
            end
            
            % Eigendecomposition 
            [V1,A1]=eig(Sb1,Sw1);          % size(V1)=[D1,D1]
            [A1,indx]=sort(diag(A1),'descend');
            V1=V1(:,indx);
            W1(:,:,iteration)=V1(:,1:L);     % size(W1)=[D1,L,iteration]
        end
        if iteration==2
            W2=[];
            Y_1=[];
            Y_11=[];
            Y_12=[];
            Mu_Y_1c=[];
        end
        if k==2
            for i=1:N
                Y_2(:,:,i)=(W1(:,:,iteration)'*X(:,:,i))';          % k=2, size(Y_2)=[D2,L,N]
            end
            for i=1:N1
                Y_21(:,:,i)=(W1(:,:,iteration)'*X1(:,:,i))';        % k=2, class=1, size(Y_21)=[D2,L,N1]
            end
            for i=1:N2
                Y_22(:,:,i)=(W1(:,:,iteration)'*X2(:,:,i))';        % k=2, class=2, size(Y_22)=[D2,L,N2]
            end
            Y_2c={Y_21,Y_22};
            Mu_Y_2=mean(Y_2,3);         % size(Mu_Y_2)=[D2,L]
            Mu_Y_21=mean(Y_21,3);       % size(Mu_Y_21)=[D2,L]
            Mu_Y_22=mean(Y_22,3);       % size(Mu_Y_22)=[D2,L]
            Mu_Y_2c(:,:,1)=Mu_Y_21;
            Mu_Y_2c(:,:,2)=Mu_Y_22;
            
            % beetween & within class scatter
            Sb2=0;
            Sw_j=0;
            Sw2=0;
            for c=1:2
                Sb=n(c)*(Mu_Y_2c(:,:,c)-Mu_Y_2)*(Mu_Y_2c(:,:,c)-Mu_Y_2)';
                Sb=(Sb+Sb')/2;
                Sb2=Sb2+Sb;           % size(Sb_2)=[D2,D2]
                for j=1:n(c)
                    Sw=(Y_2c{c}(:,:,j)-Mu_Y_2c(:,:,c))*(Y_2c{c}(:,:,j)-Mu_Y_2c(:,:,c))';
                    Sw=(Sw+Sw')/2;
                    Sw_j=Sw_j+Sw;
                end
                Sw2=Sw2+Sw_j;     % size(Sw2)=[D2,D2]
            end
            
            % Eigendecomposition
            [V2,A2]=eig(Sb2,Sw2);          % size(V2)=[D2,D2]
            [A2,indx]=sort(diag(A2),'descend');
            V2=V2(:,indx);
            W2(:,:,iteration)=V2(:,1:L);     % size(W2)=[D2,L,iteration]
        end
    end
    if iteration>2
        Err1=norm(W1(:,:,iteration)-W1(:,:,iteration-1));
        Err2=norm(W2(:,:,iteration)-W2(:,:,iteration-1));
%         err=10^(-5);
        if Err1<err && Err2<err
            break
        end
    end
end
W1_op=W1(:,:,iteration);        % optimal W1, size(W1_op)=[D1,L]
W2_op=W2(:,:,iteration);        % optimal W2, size(W2_op)=[D2,L]
for i=1:N1
    B1=(W1_op'*X1(:,:,i))*W2_op;        % size(B1)=[L,L]
    F1(:,i)=B1(:);      % size(F1)=[L^2,N1]
end
for i=1:N2
    B2=(W1_op'*X2(:,:,i))*W2_op;        % size(B2)=[L,L]
    F2(:,i)=B2(:);      % size(F2)=[L^2,N2]
end
T=0;
[w,b]=myLDA_train(F1,F2,T);     % size(w)=[L^2,1]

end

