%% asst 2 code


close all;


dataC(:,1)=textread('ClassificationX.txt','','delimiter',',');
dataC(:,2)=textread('ClassificationY.txt','','delimiter',',');
trainC=dataC(1:50,:);
valC=dataC(51:100,:);
testC=dataC(101:200,:);





[~,testError2a,confusionMatrix2a]=kNN(trainC,testC,1,'c');




valError2b=zeros(1,6);
for k=1:6
    [~,valError2b(k)]=kNN(trainC,valC,2*k-1,'c');
end
[minValError2b,minK2b]=min(valError2b);
minK2b=minK2b*2-1;
[~,testError2b,confusionMatrix2b]=kNN(trainC,testC,minK2b,'c');




valError2c=zeros(1,6);
valLikelihood2c=zeros(1,6);
for k=1:6
    [~,valError2c(k),~,valLikelihood2c(k)]=kNN(trainC,valC,2*k-1,'p');
end
[minValError2c,minK2c]=min(valError2c);
minK2c=minK2c*2-1;
[~,testError2c,~,testLikelihood2c]=kNN(trainC,testC,minK2c,'p');



testError2d=zeros(18,1);
for thresh=1:18
    [~,testError2d(thresh)]=LR(trainC,testC,1,0.2,0.999,thresh/18,'t');
end


trainLikelihood2f=zeros(10,500000);
valLikelihood2f=zeros(1,10);
for k=1:10
    [~,~,trainLikelihood2f(k,:),valLikelihood2f(k)]=LR(trainC,valC,k,0.8,0.999,0.5,'c');
end
[maxValLikelihood2f,maxK2f]=max(valLikelihood2f);
[~,testError2f,~,testLikelihood2f]=LR(trainC,testC,maxK2f,0.8,0.999,0.5,'c');





%%% asst 3


%%
trainMSEError2g1=zeros(4,10000);
testMSEError2g1=zeros(4,10000);
trainClassError2g1=zeros(4,10000);
testClassError2g1=zeros(4,10000);
trainC(:,2)
for mu=0.1:0.1:0.4
[~,trainMSEError2g1(single(10*mu),:),testMSEError2g1(single(10*mu),:),trainClassError2g1(single(10*mu),:),testClassError2g1(single(10*mu),:)]=NN(trainC,testC,5,mu,0,10000,0.2*rand(2*5,1)-0.1,'t');
mu
end
trainMSEError2g1
figure();
semilogx(1:10000,trainMSEError2g1(1,:),'k',1:10000,trainMSEError2g1(2,:),'b',1:10000,trainMSEError2g1(3,:),'r',1:10000,trainMSEError2g1(4,:),'m');
legend('Learning rate: 0.1','Learning rate: 0.2','Learning rate: 0.3','Learning rate: 0.4');
ylabel('Training MSE');
xlabel('Epochs');
grid on;
hold off;
figure();
semilogx(1:10000,trainClassError2g1(1,:),'k',1:10000,trainClassError2g1(2,:),'b',1:10000,trainClassError2g1(3,:),'r',1:10000,trainClassError2g1(4,:),'m');
legend('Learning rate: 0.1','Learning rate: 0.2','Learning rate: 0.3','Learning rate: 0.4');
ylabel('Training classification error');
xlabel('Epochs');
grid on;
hold off;
%
%%%
%trainMSEError2g2=zeros(20,10000);
%testMSEError2g2=zeros(20,10000);
%trainClassError2g2=zeros(20,10000);
%testClassError2g2=zeros(20,10000);
%for i=1:20
%[~,trainMSEError2g2(i,:),testMSEError2g2(i,:),trainClassError2g2(i,:),testClassError2g2(i,:)]=NN(trainC,testC,5,0.2,0,10000,0.2*rand(2*5,1)-0.1,'t');
%end
%figure();
%hist(trainMSEError2g2(:,end));
%ylabel('Frequency');
%xlabel('Training MSE');
%hold off;
%figure();
%hist(trainClassError2g2(:,end));
%ylabel('Frequency');
%xlabel('Training classification error');
%hold off;
%[minVal2g2,minCase2g2]=min(trainMSEError2g2(:,end));
%
%%%
%valMSEError2g3=zeros(1,10);
%valClassError2g3=zeros(1,10);
%for k=1:10
%    MSEerror2g3=zeros(20,100);
%    classError2g3=zeros(20,100);
%    for i=1:20
%        [~,~,MSEerror2g3(i,:),~,classError2g3(i,:)]=NN(trainC,valC,k,0.2,0,100,0.2*rand(2*k,1)-0.1,'t');
%    end
%    [valMSEError2g3(k),ind]=min(MSEerror2g3(:,end));
%    valClassError2g3(k)=classError2g3(ind,end);    
%    k
%end
%[minValMSEError2g3,minK2g3]=min(valMSEError2g3);
%[~,~,testMSEError2g3,~,testClassError2g3]=NN(trainC,testC,minK2g3,0.2,0,100,0.2*rand(2*minK2g3,1)-0.1,'t');
%
%%%
%[~,trainMSEError2g4,valMSEError2g4,trainClassError2g4,valClassError2g4,]=NN(trainC,valC,10,0.2,0,10000,0.2*rand(2*10,1)-0.1,'t');
%figure();
%semilogx(1:10000,trainMSEError2g4,'k',1:10000,valMSEError2g4,'r');
%legend('Training MSE','Validation MSE');
%ylabel('Error');
%xlabel('Epochs');
%hold off;
%figure();
%semilogx(1:10000,trainClassError2g4,'k',1:10000,valClassError2g4,'r');
%legend('Training classification error','Validation classification error');
%ylabel('Error');
%xlabel('Epochs');
%hold off;
%numEpochs2g4=7;
%[~,trainMSEError2g4,testMSEError2g4,trainClassError2g4,testClassError2g4]=NN(trainC,testC,10,0.2,0,numEpochs2g4,0.2*rand(2*10,1)-0.1,'t');
%
%%%
%trainMSEError2h1=zeros(4,10000);
%testMSEError2h1=zeros(4,10000);
%trainClassError2h1=zeros(4,10000);
%testClassError2h1=zeros(4,10000);
%winit=0.2*rand(2*5,1)-0.1;
%for mu=0.1:0.1:0.4
%    [~,trainMSEError2h1(single(10*mu),:),testMSEError2h1(single(10*mu),:),trainClassError2h1(single(10*mu),:),testClassError2h1(single(10*mu),:)]=NN(trainC,testC,5,mu,0,10000,winit,'c');
%end
%figure();
%semilogx(1:10000,trainMSEError2h1(1,:),'k',1:10000,trainMSEError2h1(2,:),'b',1:10000,trainMSEError2h1(3,:),'r',1:10000,trainMSEError2h1(4,:),'m');
%legend('Learning rate: 0.1','Learning rate: 0.2','Learning rate: 0.3','Learning rate: 0.4');
%ylabel('Training log-likelihood');
%xlabel('Epochs');
%hold off;
%figure();
%semilogx(1:10000,trainClassError2h1(1,:),'k',1:10000,trainClassError2h1(2,:),'b',1:10000,trainClassError2h1(3,:),'r',1:10000,trainClassError2h1(4,:),'m');
%legend('Learning rate: 0.1','Learning rate: 0.2','Learning rate: 0.3','Learning rate: 0.4');
%ylabel('Training classification error');
%xlabel('Epochs');
%hold off;
%
%%%
%trainMSEError2h2=zeros(20,10000);
%testMSEError2h2=zeros(20,10000);
%trainClassError2h2=zeros(20,10000);
%testClassError2h2=zeros(20,10000);
%for i=1:20
%[~,trainMSEError2h2(i,:),testMSEError2h2(i,:),trainClassError2h2(i,:),testClassError2h2(i,:)]=NN(trainC,testC,5,0.4,0,10000,0.2*rand(2*5,1)-0.1,'c');
%end
%figure();
%hist(trainMSEError2h2(:,end));
%ylabel('Frequency');
%xlabel('Training log-likelihood');
%hold off;
%figure();
%hist(trainClassError2h2(:,end));
%ylabel('Frequency');
%xlabel('Training classification error');
%hold off;
%[maxVal2h2,maxCase2h2]=max(trainMSEError2h2(:,end));
%
%%%
%valMSEError2h3=zeros(1,10);
%valClassError2h3=zeros(1,10);
%for k=1:10
%    MSEerror2h3=zeros(20,100);
%    classError2h3=zeros(20,100);
%    for i=1:20
%        [~,~,MSEerror2h3(i,:),~,classError2h3(i,:)]=NN(trainC,valC,k,0.4,0,100,0.2*rand(2*k,1)-0.1,'c');
%    end
%    [valMSEError2h3(k),ind]=max(MSEerror2h3(:,end));
%    valClassError2h3(k)=classError2h3(ind,end);    
%    k
%end
%[minValMSEError2h3,minK2h3]=max(valMSEError2h3);
%[~,~,testMSEError2h3,~,testClassError2h3]=NN(trainC,testC,minK2h3,0.4,0,100,0.2*rand(2*minK2h3,1)-0.1,'c');
%
%%%
%[~,trainMSEError2h4,valMSEError2h4,trainClassError2h4,valClassError2h4,]=NN(trainC,valC,10,0.4,0,10000,0.2*rand(2*10,1)-0.1,'c');
%figure();
%semilogx(1:10000,trainMSEError2h4,'k',1:10000,valMSEError2h4,'r');
%legend('Training log-likelihood','Validation log-likelihood');
%ylabel('Error');
%xlabel('Epochs');
%hold off;
%figure();
%semilogx(1:10000,trainClassError2h4,'k',1:10000,valClassError2h4,'r');
%legend('Training classification error','Validation classification error');
%ylabel('Error');
%xlabel('Epochs');
%hold off;
%numEpochs2h4=40;
%[~,trainMSEError2h4,testMSEError2h4,trainClassError2h4,testClassError2h4]=NN(trainC,testC,10,0.4,0,numEpochs2h4,0.2*rand(2*10,1)-0.1,'c');
%
%%%
%tic;
%subset2i=zeros(4,13);
%for i=1:4
%    subset2i(i,:)=arrayfun(@(x) str2num(x), dec2bin(floor(2^13.*rand(1,1)),13));
%end
%[~,testError2i]=ensembl(trainC,testC,minK2b,minK2c,maxK2f,minK2g3,0.2,0.2*rand(2*5,1)-0.1,0.2*rand(2*minK2g3,1)-0.1,0.2*rand(2*10,1)-0.1,numEpochs2g4,minK2h3,0.4,0.2*rand(2*5,1)-0.1,0.2*rand(2*minK2h3,1)-0.1,0.2*rand(2*10,1)-0.1,numEpochs2h4,subset2i);
%
%toc
%
%%%
%range=linspace(floor(min(dataC(:,1))),ceil(max(dataC(:,1))),100);
%figure();
%plot(trainC(:,1),trainC(:,2),'+k');
%hold on;
%plot(testC(:,1),testC(:,2),'ok');
%hold on;
%[predictedOut2a]=kNN(trainC,[range' round(rand(100,1))],1,'c');
%plot(range,predictedOut2a,'k');
%hold on;
%[predictedOut2b]=kNN(trainC,[range' round(rand(100,1))],minK2b,'c');
%plot(range,predictedOut2b,'b');
%hold on;
%[predictedOut2c]=kNN(trainC,[range' round(rand(100,1))],minK2c,'p');
%plot(range,predictedOut2c,'r');
%hold on;
%[predictedOut2d]=LR(trainC,[range' round(rand(100,1))],1,0.2,0.999,0.5,'t');
%plot(range,predictedOut2d,'g');
%hold on;
%[predictedOut2f]=LR(trainC,[range' round(rand(100,1))],maxK2f,0.8,0.999,0.5,'c');
%plot(range,predictedOut2f,'m');
%hold on;
%
%
%
%[predictedOut2g1]=NN(trainC,[range' rand(100,1)],5,0.2,0,10000,0.2*rand(2*5,1)-0.1,'t');
%plot(range,predictedOut2g1,'c');
%hold on;
%[predictedOut2g2]=NN(trainC,[range' rand(100,1)],5,0.2,0,10000,0.2*rand(2*5,1)-0.1,'t');
%plot(range,predictedOut2g2,'k--');
%hold on;
%[predictedOut2g3]=NN(trainC,[range' rand(100,1)],minK2g3,0.2,0,10000,0.2*rand(2*minK2g3,1)-0.1,'t');
%plot(range,predictedOut2g3,'b--');
%hold on;
%[predictedOut2g4]=NN(trainC,[range' rand(100,1)],10,0.2,0,numEpochs2g4,0.2*rand(2*10,1)-0.1,'t');
%plot(range,predictedOut2g4,'r--');
%hold on;
%
%
%[predictedOut2h1]=NN(trainC,[range' rand(100,1)],5,0.4,0,10000,0.2*rand(2*5,1)-0.1,'c');
%plot(range,predictedOut2h1,'g--');
%hold on;
%[predictedOut2h2]=NN(trainC,[range' rand(100,1)],5,0.4,0,10000,0.2*rand(2*5,1)-0.1,'c');
%plot(range,predictedOut2h2,'y--');
%hold on;
%[predictedOut2h3]=NN(trainC,[range' rand(100,1)],minK2h3,0.4,0,10000,0.2*rand(2*minK2h3,1)-0.1,'c');
%plot(range,predictedOut2h3,'m--');
%hold on;
%[predictedOut2h4]=NN(trainC,[range' rand(100,1)],10,0.4,0,numEpochs2h4,0.2*rand(2*10,1)-0.1,'c');
%plot(range,predictedOut2h4,'c--');
%hold on;
%
%
%[predictedOut2i]=ensembl(trainC,testC,minK2b,minK2c,maxK2f,minK2g3,0.2,0.2*rand(2*5,1)-0.1,0.2*rand(2*minK2g3,1)-0.1,0.2*rand(2*10,1)-0.1,numEpochs2g4,minK2h3,0.4,0.2*rand(2*5,1)-0.1,0.2*rand(2*minK2h3,1)-0.1,0.2*rand(2*10,1)-0.1,numEpochs2h4,subset2i);
%
%plot(range,predictedOut2i(:,4),'k-.');
%hold on;
%
%legend('Training Data','Test Data','2-a','2-b','2-c','2-d','2-e','3-a','3-b','3-c','3-d','3-e1','3-e2','3-e3','3-e4','ensembl');
%
%xlabel('Input');
%ylabel('Prediction');
%ylim([-0.5 1.5]);
%hold off;
%
%
%
