function [ Y ] = myLDA_class( X,w,b )
%Linear Discriminant Analysis classifier
%   X:test data(F*N)
%   w:projection vector(F*1)
%   b:bias term(a number)
%   Y:label of X

%% Traditional LDA Classifier
Y=[];
for i=1:size(X,2)
    if (w'*X(:,i))+b>0
        y=1;
    else
        y=0;
    end
    Y=[Y,y];
end

end

