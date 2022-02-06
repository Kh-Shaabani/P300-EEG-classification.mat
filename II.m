clc
clear
[pathstr,name,ext]=fileparts(which(mfilename));
adrs=strfind(pathstr,'\Simulation');
Subjects = ('AB');
N_trainchar = 40;       % number of training characters
N_testchar = size(Signal,1)-N_trainchar;        % number of test characters
j_trainchar = N_trainchar;
N_RowsColumns = 12;
N_trials = 15;
for i_sbj=1:length(Subjects)
    d1=[erase(pathstr,pathstr(adrs:end)),'\BCI_Comp_III_Wads_2004\Subject_',Subjects(i_sbj),'_Train.mat'];
    load(d1);
    StimulusType=double(StimulusType);
    StimulusCode=double(StimulusCode);
    Flashing=double(Flashing);
    Signal=double(Signal);
    Fs = 240;
    T_on = 0.1;
    T_off = 0.075;
    Fs_on = Fs*T_on;
    Fs_off = Fs*T_off;
    ds = 6;     % downsample to 40 Hz
    window = 168;
    CHANNELS = {'FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5',...
        'CP3','CP1','CPz','CP2','CP4','CP6','Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F7',...
        'F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FT8','T7','T8','T9','T10','TP7','TP8','P7',...
        'P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'};
    Ch = [32 34 36 41 9 11 13 42 47 49 51 53 55 56 60 62];
    channels = CHANNELS(Ch);
    [b,a]=butter(3,[0.1 30]/Fs/2,'bandpass');       % butterworth bandpass filter
    Data=[];
    
    %% for each character epoch (Block)
    for epoch=1:size(Signal,1)
        % get reponse samples at start of each Flash
        rowcolcnt=zeros(1,12);
        rowcolcnt1=zeros(1,2);
        rowcolcnt2=zeros(1,10);
        i_1=0;
        i_2=0;
        
        % Data Extraction & Segmentation (Windowing)
        for n=2:size(Signal,2)
            if Flashing(epoch,n)==0 && Flashing(epoch,n-1)==1
                rowcol=StimulusCode(epoch,n-1);
                rowcolcnt(rowcol)=rowcolcnt(rowcol)+1;
                X=Signal(epoch,(n-Fs_on):(n+window-Fs_on-1),Ch);     % size(X)=[1,168,16]
                
                % Preprocessing
                X=squeeze(X);       % size(X)=[168,16]
                X=filtfilt(b,a,X);        % filter, size(X)=[168,16]
%                 for i=1:size(X,2)
%                     X=decimate(X(:,i),ds);
%                     X(:,i)=X;
%                 end
                X=X(1:ds:end,:);      % downsample, size(X)=[28,16]
                
                responses(rowcol,rowcolcnt(rowcol),:,:,epoch)=X;     % size(responses)=[12,15,28,16,85]
                % responses(1)=rowcol=number of rows & columns=12
                % responses(2)=rowcolcnt(rowcol)=number of trials per run=15
                % responses(3)=n-24:n+window-25=number of downsamples in each window=28
                % responses(4)=:=number of channels=16
                % responses(5)=:=number of channels=85
                
                %% Seperate data of each class
                if StimulusType(epoch,n-1)==1
                    i_1=i_1+1;
                    rowcol1(i_1)=StimulusCode(epoch,n-1);
                    if i_1==1
                        j_1=1;
                    else
                        for m=1:i_1-1
                            if rowcol1(i_1)==rowcol1(m)
                                j_1=m;
                                break
                            else
                                j_1=m+1;
                            end
                        end
                    end
                    rowcolcnt1(j_1)=rowcolcnt1(j_1)+1;
                    class1(j_1,rowcolcnt1(j_1),:,:)=X;     % size(class1)=[2,15,28,16]    % class1:target
                else
                    i_2=i_2+1;
                    rowcol2(i_2)=StimulusCode(epoch,n-1);
                    if i_2==1
                        j_2=1;
                    else
                        for m=1:i_2-1
                            if rowcol2(i_2)==rowcol2(m)
                                j_2=m;
                                break
                            else
                                j_2=m+1;
                            end
                        end
                    end
                    rowcolcnt2(j_2)=rowcolcnt2(j_2)+1;
                    class2(j_2,rowcolcnt2(j_2),:,:)=X;     % size(class2)=[10,15,28,16]    % class2:nontarget
                end
            end
        end
        Data.Epoch(epoch).Target=class1;
        Data.Epoch(epoch).NonTarget=class2;
    end
    
    
    %% Training & Test
    class1_train=[];    class1_test=[];
    class2_train=[];    class2_test=[];
    for i=1:N_trainchar
        class1_train(:,:,:,:,i)=Data.Epoch(i).Target;          % size(class1_train)=[2,15,28,16,40]
        class2_train(:,:,:,:,i)=Data.Epoch(i).NonTarget;       % size(class2_train)=[10,15,28,16,40]
    end
%     for i=N_trainchar+1:size(Data.Epoch,2)
%         class1_test(:,:,:,i-N_trainchar)=squeeze(mean(Data.Epoch(i).Target,2));   % size(class1_test)=[2,28,16,45]
%         class2_test(:,:,:,i-N_trainchar)=squeeze(mean(Data.Epoch(i).NonTarget,2));% size(class2_test)=[10,28,16,45]
%     end
    
    
%     for j_trainchar=3:N_trainchar
        %% LDA
        % LDA train
        % class1_train
        A=[];
        cnt=0;
        for i=1:size(class1_train,3)
            for j=1:size(class1_train,4)
                cnt=cnt+1;
                A(cnt,:,:,:)=squeeze(class1_train(:,:,i,j,:));      % size(A)=[448,2,15,40]
            end
        end

        train1=[];
        cnt=0;
        for i=1:size(A,2)
            for j=1:size(A,3)
                for k=1:j_trainchar
                    cnt=cnt+1;
                    train1(:,cnt)=squeeze(A(:,i,j,k));      % size(train1)=[448,1200]=[feature,sample]
                end
            end
        end

        % class2_train
        A=[];
        cnt=0;
        for i=1:size(class2_train,3)
            for j=1:size(class2_train,4)
                cnt=cnt+1;
                A(cnt,:,:,:)=squeeze(class2_train(:,:,i,j,:));      % size(A)=[448,10,15,40]
            end
        end

        train2=[];
        cnt=0;
        for i=1:size(A,2)
            for j=1:size(A,3)
                for k=1:j_trainchar
                    cnt=cnt+1;
                    train2(:,cnt)=squeeze(A(:,i,j,k));      % size(train2)=[448,6000]=[feature,sample]
                end
            end
        end
        
        T=0;
        [ w_LDA,~ ] = myLDA_train( train1,train2,T );     % myLDA_tarin
        
        % LDA test
        % test data
        test=[];
        resp=squeeze(mean(responses,2));     % size(resp)=[12,28,16,85]
        for m=N_trainchar+1:size(resp,4)
            cnt=0;
            for i=1:size(resp,2)
                for j=1:size(resp,3)
                    cnt=cnt+1;
                    test(cnt,:,m-N_trainchar)=squeeze(resp(:,i,j,m));      % size(test)=[448,12,45]
                end
            end
        end

        [ label_RC_p,label_Char_p ] = myLDA_testII( test,w_LDA );     % myLDA_test
        % predicted lables of target row & column, size(label_RC_p)=[2,45]
        % predicted lables of target character, size(label_Char_p)=[1,45]
        
        label_RC=[];
        for i=N_trainchar+1:size(StimulusType,1)
            l=unique(StimulusCode(i,:).*StimulusType(i,:));     % size(l)=[1,3], label=[0,column,row]
            l(1)=[];
            label_RC=[label_RC,l'];     % true lables of target row & column, size(label)=[2,45]
            label_Char(:,i-N_trainchar)=(6*(label_RC(2,i-N_trainchar)-7))+label_RC(1,i-N_trainchar);
            % true lables of target characters, size(targetlabel)=[1,45]
        end
        
        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_LDA(i_sbj)=100*true/(2*45);
        
        % ROC
        labels_T=[];        % True labels
        for i=1:size(label_RC,2)
            lbl_T=zeros(1,12);
            lbl_T(label_RC(:,i))=1;         % size(lbl_T)=[1,12]
            labels_T=[labels_T,lbl_T];      % size(labels_T)=[1,540]
        end
        
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,12);
            lbl_P(label_RC_p(:,i))=1;       % size(lbl_P)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(labels_P)=[1,540]
        end
        
        [~,~,~,AUC_LDA(i_sbj)] = perfcurve(labels_T,labels_P,1);
%     end
end

