function [ w,b,SW_Features ] = mySWLDA_train( X1,X2,y,T )
%StepWise Linear Discriminant Analysis training for two classes
%   Xi:matrix of training data of class i, size(Xi)=[D,Ni]=[feature,sample].
%   y:ordered class labels, size(y)=[N,1] that N is the number of total training samples.
%   T:threshold.
%   w:projection vector(L*1), (L<<D).
%   b:bias term.
%   SW_Features:Indices of SWLDA features,size(SW_Features)=[1,L]

X=[X1';X2'];     % size(X)=[N,D]
[~,~,~,inmodel]=stepwisefit(X,y,'penter',0.1,'premove',0.15,'display','off');
% size(inmodel)=[1,D], 1 for features(columns) that are in the model.
SW_Features=find(inmodel==1);       % size(SW_Features)=[1,L]

% mdl=stepwiselm(C,y,'linear');

train1_SW=X1(SW_Features,:);        % size(train1_SW)=[L,N1]
train2_SW=X2(SW_Features,:);        % size(train2_SW)=[L,N2]
[ w,b ] = myLDA_train( train1_SW,train2_SW,T );     % size(w)=[L,1]

end

