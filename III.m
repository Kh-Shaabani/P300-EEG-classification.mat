clc
clear
tic
[pathstr,name,ext]=fileparts(which(mfilename));
adrs=strfind(pathstr,'\Simulation');
Subjects = ('AB');
N_trainchar = 40;       % number of training characters
N_testchar = size(Signal,1)-N_trainchar;        % number of test characters
% j_trainchar = N_trainchar;
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
    Data = [];
        
    %% for each character epoch (Block)
    for epoch=1:size(Signal,1) 
        % get response samples at start of each Flash
        rowcolcnt=zeros(1,12);
        rowcolcnt1=zeros(1,2);      % for targets
        rowcolcnt2=zeros(1,10);     % for nontargets
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
                X=filtfilt(b,a,X);        % size(X)=[168,16]
%                 for i=1:size(X,2)
%                     X=decimate(X(:,i),ds);
%                     X(:,i)=X;
%                 end
                X=X(1:ds:end,:);      % downsample, size(X)=[28,16]
            
                responses(rowcol,rowcolcnt(rowcol),:,:,epoch)=X;     % size(responses)=[12,15,28,16,85]
                % responses(1)=rowcol=number of rows & columns=12
                % responses(2)=rowcolcnt(rowcol)=number of trials per run=15
                % responses(3)=n-24:n+window-25=number of samples in each window=28
                % responses(4)=:=number of channels=16
                % responses(5)=:=number of epochs=85
                    
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
                    class1(j_1,rowcolcnt1(j_1),:,:)=X;     % size(class1)=[2,15,28,16]    % class1:target=1
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
                    class2(j_2,rowcolcnt2(j_2),:,:)=X;     % size(class2)=[10,15,28,16]    % class2:nontarget=0
                end
            end
        end
        Data.Epoch(epoch).Target=class1;
        Data.Epoch(epoch).NonTarget=class2;
    end
    
    
    %% Training & Test
    class1_train = [];    class1_test = [];
    class2_train = [];    class2_test = [];
    for i=1:N_trainchar
        class1_train(:,:,:,:,i)=Data.Epoch(i).Target;          % size(class1_train)=[2,15,28,16,40]
        class2_train(:,:,:,:,i)=Data.Epoch(i).NonTarget;       % size(class2_train)=[10,15,28,16,40]
    end
%     for i=N_trainchar+1:size(Data.Epoch,2)
%         class1_test(:,:,:,:,i-N_trainchar)=Data.Epoch(i).Target;        % size(class1_test)=[2,15,28,16,45]
%         class2_test(:,:,:,:,i-N_trainchar)=Data.Epoch(i).NonTarget;     % size(class2_test)=[10,15,28,16,45]
%     end


    %% Grand Mean of Targets and NonTargets on subjects & on channels in windows
    % class1_train (Targets)
%     A=[];
%     cnt=0;
%     for i=1:size(class1_train,5)
%         for j=1:size(class1_train,1)
%             for k=1:size(class1_train,2)
%                 cnt=cnt+1;
%                 A(:,:,cnt)=squeeze(class1_train(j,k,:,:,i));      % size(A)=[28,16,1200]
%             end
%         end
%     end
%     B=mean(A,3);        % size(B)=[28,16]
%     y1=mean(B,2);       % size(y1)=[28,1]
%     figure(1)
%     title('Grand Mean')
%     plot(y1,'g')
%     xlabel('Time Samples')
%     ylabel('P300 Amplitude')
    
    % class2_train (NonTargets)
%     A=[];
%     cnt=0;
%     for i=1:size(class2_train,5)
%         for j=1:size(class2_train,1)
%             for k=1:size(class2_train,2)
%                 cnt=cnt+1;
%                 A(:,:,cnt)=squeeze(class2_train(j,k,:,:,i));      % size(A)=[28,16,6000]
%             end
%         end
%     end
%     B=mean(A,3);        % size(B)=[28,16]
%     y2=mean(B,2);       % size(y2)=[28,1]
%     hold on
%     plot(y2,'r');
%     legend('Targets','NonTargets')
        
        
    for j_trainchar=3:N_trainchar
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
%                 for k=1:size(A,4)     % for Grand Mean
                    cnt=cnt+1;
                    train1(:,cnt)=squeeze(A(:,i,j,k));      % size(train1)=[448,1200]=[feature,sample]
                end
            end
        end
        
        % Mean of Targets on subjects in features (time samples,channels)
%         y1=mean(train1,2);        % sixe(y1)=[448,1]
%         figure(2)
%         title('Grand Mean')
%         plot(y1,'g');
%         xlabel('Feature Samples')
%         ylabel('P300 Amplitude')
        %
        
        % class2_train
        A=[];
        cnt=0;
        for i=1:size(class2_train,3)
            for j=1:size(class2_train,4)
                cnt=cnt+1;
                A(cnt,:,:,:)=squeeze(class2_train(:,:,i,j,:));      % size(B)=[448,10,15,40]
            end
        end
        
        train2=[];
        cnt=0;
        for i=1:size(A,2)
            for j=1:size(A,3)
                for k=1:j_trainchar
%                 for k=1:size(A,4)     % for Grand Mean
                    cnt=cnt+1;
                    train2(:,cnt)=squeeze(A(:,i,j,k));      % size(train2)=[448,6000]=[feature,sample]
                end
            end
        end
        
        % Grand Mean of NonTargets on subjects in features (time samples,channels)
%         hold on
%         y2=mean(train2,2);        % sixe(y2)=[448,1]
%         plot(y2,'r');
%         legend('Targets','NonTargets')
        %
        
        T=0;
        [ w_LDA,~ ] = myLDA_train( train1,train2,T );     % myLDA_tarin
        
        % LDA test
        % test data
        test=[];
        for m=N_trainchar+1:size(responses,5)
            cnt=0;
            for i=1:size(responses,3)
                for j=1:size(responses,4)
                    cnt=cnt+1;
                    test(cnt,:,:,m-N_trainchar)=squeeze(responses(:,:,i,j,m));  % size(test)=[448,12,15,45]
                end
            end
        end
        
        [ label_RC_p,label_Char_p ] = myLDA_testIII( test,w_LDA );     % myLDA_test
        % predicted lables of target row & column, size(label_RC_p)=[2,45]
        % predicted lables of target character, size(label_Char_p)=[1,45]
        
        label_RC=[];
        for i=N_trainchar+1:size(StimulusType,1)
            l=unique(StimulusCode(i,:).*StimulusType(i,:));     % size(l)=[1,3], l=[0,column,row]
            l(1)=[];
            label_RC=[label_RC,l'];     % true lables of target row & column, size(label_RC)=[2,45]
            label_Char(:,i-N_trainchar)=(6*(label_RC(2,i-N_trainchar)-7))+label_RC(1,i-N_trainchar);
            % true lables of target characters, size(label_Char)=[1,45]
        end
        
        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_LDA=100*true/(2*(size(Signal,1)-N_trainchar));
        
        % ROC
        labels_T=[];        % True labels
        for i=1:size(label_RC,2)
            lbl_T=zeros(1,N_RowsColumns);
            lbl_T(label_RC(:,i))=1;         % size(lbl_T)=[1,12]
            labels_T=[labels_T,lbl_T];      % size(labels_T)=[1,N_testblock*N_buttons]=[1,540=12*45]
        end
        
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,N_RowsColumns);
            lbl_P(label_RC_p(:,i))=1;       % size(lbl_P)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(labels_P)=[1,N_testblock*N_buttons]=[1,540=12*45]
        end
        
        [~,~,~,AUC_LDA] = perfcurve(labels_T,labels_P,1);
        
        
        %% SKLDA
        % SKLDA train
        [ w_SK,~ ] = mySKLDA_train( train1,train2,T );      % size(w_SK)=[448,1]
        
        % SKLDA test
        [ label_RC_p,label_Char_p ] = myLDA_testIII( test,w_SK );
        % size(label_RC_p)=[2,45], % size(label_Char_p)=[1,45]
        
        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_SKLDA=100*true/(2*(size(Signal,1)-N_trainchar));
        
        % ROC
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,N_RowsColumns);
            lbl_P(label_RC_p(:,i))=1;       % size(lbl_P)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(labels_P)=[1,540]
        end
        
        [~,~,~,AUC_SKLDA] = perfcurve(labels_T,labels_P,1);
        
        
        %% SWLDA
        % SWLDA train
        y=[ones(size(train1,2),1);zeros(size(train2,2),1)];     % size(y)=[7200=12*15*40,1]
        [ w_SW,~,SW_Features ] = mySWLDA_train( train1,train2,y,T );% size(w_SW)=[51,1], size(SW_Features)=[1,51]
        
        % SWLDA test
        [ label_RC_p,label_Char_p ] = myLDA_testIII( test(SW_Features,:,:,:),w_SW );
        % size(label_RC_p)=[2,45], size(label_Char_p)=[1,45]
        
        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_SWLDA=100*true/(2*(size(Signal,1)-N_trainchar));
        
        % ROC
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,N_RowsColumns);
            lbl_P(label_RC_p(:,i))=1;       % size(lbl_P)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(labels_P)=[1,540]
        end
        
        [~,~,~,AUC_SWLDA] = perfcurve(labels_T,labels_P,1);
        
        
        %% STDA
        % STDA train
        % class1_train
        train1_ST=[];
        for i=1:size(class1_train,4)
            cnt=0;
            for j=1:size(class1_train,1)
                for k=1:size(class1_train,2)
                    for l=1:j_trainchar
                        cnt=cnt+1;
                        train1_ST(i,:,cnt)=class1_train(j,k,:,i,l);      % size(train1_ST)=[16,28,1200]
                    end
                end
            end
        end
        
        % class2_train
        train2_ST=[];
        for i=1:size(class2_train,4)
            cnt=0;
            for j=1:size(class2_train,1)
                for k=1:size(class2_train,2)
                    for l=1:j_trainchar
                        cnt=cnt+1;
                        train2_ST(i,:,cnt)=class2_train(j,k,:,i,l);      % size(train2_ST)=[16,28,6000]
                    end
                end
            end
        end
        
        L=2;
        err=10^(-5);
        [ w_ST,~,W1_op,W2_op,iteration ] = mySTDA_train( train1_ST,train2_ST,L,err );       % size(w_ST)=[4,1]
        
        % STDA test
        test_ST=[];
        for i=1:size(responses,4)
            for j=1:size(responses,3)
                for k=1:size(responses,1)
                    for l=1:size(responses,2)
                        for m=N_trainchar+1:size(responses,5)
                            test_ST(i,j,k,l,m-N_trainchar)=responses(k,l,j,i,m); % size(test_ST)=[16,28,12,15,45]
                        end
                    end
                end
            end
        end
        
        [ label_RC_p,label_Char_p ] = mySTDA_test( test_ST,W1_op,W2_op,w_ST );
        % size(label_RC_p)=[2,45], % size(label_Char_p)=[1,45]
        
        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_STDA=100*true/(2*(size(Signal,1)-N_trainchar));
        
        % ROC
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,12);
            lbl_P(label_RC_p(:,i))=1;       % size(lbl_P)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(labels_P)=[1,540]
        end
        
        [~,~,~,AUC_STDA] = perfcurve(labels_T,labels_P,1);
        
        
        %% FC+LDA
        % FC+LDA train
        % FC train
        train1_FC=train1_ST;        % size(train1_FC)=[16,28,1200]
        train2_FC=train2_ST;        % size(train2_FC)=[16,28,6000]
        L=1;
        [ w_FC,~,~,Y1_FC,Y2_FC ] = myFC_train( train1_FC,train2_FC,L );
        % size(w_FC)=[16,1], size(Y1_FC)=[28,1200], size(Y2_FC)=[28,6000]
        
        % LDA train
        [ w_FC_LDA,~ ] = myLDA_train( Y1_FC,Y2_FC,T );
        
        % FC+LDA test
        % FC test
        test_FC=test_ST;        % size(test_FC)=[16,28,12,15,45]
        [ ~,Y_FC ] = myFC_test( test_FC,w_FC );     % size(Y_FC)=[28,12,15,45]
        
        % LDA test
        [ label_RC_p,label_Char_p ] = myLDA_testIII( Y_FC,w_FC_LDA );
        % size(label_RC_p)=[2,45], % size(label_Char_p)=[1,45]
        
        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_FC_LDA=100*true/(2*(size(Signal,1)-N_trainchar));
        
        % ROC
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,12);
            lbl_P(label_RC_p(:,i))=1;       % size(lbl_P)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(labels_P)=[1,540]
        end
        
        [~,~,~,AUC_FC_LDA] = perfcurve(labels_T,labels_P,1);
        
        
        %% CSP+LDA
        % CSP+LDA train
        % CSP train
        train1_CSP=train1_ST;       % size(train1_CSP)=[16,28,1200]
        train2_CSP=train2_ST;       % size(train2_CSP)=[16,28,6000]
        m=1;
        [ W_CSP,~,~,Y1_CSP,Y2_CSP ] = myCSP_train( train1_CSP,train2_CSP,m );
        % size(W_CSP)=[D1,2m], size(Y1_CSP)=[L*D2,N1], size(Y2_CSP)=[L*D2,N2]

        % LDA train
        [ w_CSP_LDA,~ ] = myLDA_train( Y1_CSP,Y2_CSP,T );       % size(w_CSP_LDA)=[2m*28]=[56,1]

        % CSP+LDA test
        % CSP test
        test_CSP=test_ST;
        [ ~,Y_CSP ] = myCSP_test( test_CSP,W_CSP );     % size(Y_CSP)=[56,12,15,45]

        % LDA test
        [ label_RC_p,label_Char_p ] = myLDA_testIII( Y_CSP,w_CSP_LDA );

        % Accuracy
        true=numel(find(label_RC_p==label_RC));
        Acc_CSP_LDA=100*true/(2*(size(Signal,1)-N_trainchar));

        % ROC
        labels_P=[];        % Predicted labels
        for i=1:size(label_RC_p,2)
            lbl_P=zeros(1,12);
            lbl_P(label_RC_p(:,i))=1;       % size(P_lbl)=[1,12]
            labels_P=[labels_P,lbl_P];      % size(P_labels)=[1,540]
        end

        [~,~,~,AUC_CSP_LDA] = perfcurve(labels_T,labels_P,1);
        
        
        %% Total Accuracy & AUC
        Acc(:,j_trainchar,i_sbj)=[Acc_LDA,Acc_SWLDA,Acc_SKLDA,Acc_CSP_LDA,Acc_FC_LDA,Acc_STDA];% size(Acc)=[6,40,2]
        AUC(:,j_trainchar,i_sbj)=[AUC_LDA,AUC_SWLDA,AUC_SKLDA,AUC_CSP_LDA,AUC_FC_LDA,AUC_STDA];% size(AUC)=[6,40,2]
    end
end

%% Plot
x=[3,5,10,15,20,25,30,35,40];
% figure1, AUC of Subject A
plot(x,AUC(1,x,1),'-ob','linewidth',1.5);
hold on, plot(x,AUC(2,x,1),'-h','color',[1,0.5,0.2],'linewidth',1.5);
hold on, plot(x,AUC(3,x,1),'-sg','linewidth',1.5);
hold on, plot(x,AUC(4,x,1),'->m','linewidth',1.5);
hold on, plot(x,AUC(5,x,1),'-kd','linewidth',1.5);
hold on, plot(x,AUC(6,x,1),'-^r','linewidth',1.5);
title('Subject A');
xlabel('Number of training characters');
ylabel('Area under the ROC curve');
legend('LDA','SWLDA','SKLDA','CSP+LDA','FC+LDA','STDA','location','southeast');
grid on;
xticks([0,3,5,10,15,20,25,30,35,40]);
yticks(0.4:0.05:1);
xlim([0 43]);
ylim([0.4 1]);

% figure2, AUC of Subject B
figure, plot(x,AUC(1,x,2),'-ob','linewidth',1.5);
hold on, plot(x,AUC(2,x,2),'-h','color',[1,0.5,0.2],'linewidth',1.5);
hold on, plot(x,AUC(3,x,2),'-sg','linewidth',1.5);
hold on, plot(x,AUC(4,x,2),'->m','linewidth',1.5);
hold on, plot(x,AUC(5,x,2),'-kd','linewidth',1.5);
hold on, plot(x,AUC(6,x,2),'-^r','linewidth',1.5);
grid on
title('Subject B');
xlabel('Number of training characters');
ylabel('Area under the ROC curve');
legend('LDA','SWLDA','SKLDA','CSP+LDA','FC+LDA','STDA','location','southeast');
grid on;
xticks([0,3,5,10,15,20,25,30,35,40,43]);
yticks(0.4:0.05:1);
xlim([0 43]);
ylim([0.4 1]);
tt=toc;
ffffff

   
   