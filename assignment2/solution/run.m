
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
figure();
plot((1:1:18)./18,testError2d,'k');
axis([0.1 0.9 0.2 0.7]);
xlabel('Threshold');
ylabel('Test error rate');
hold off;


trainLikelihood2f=zeros(10,500000);
valLikelihood2f=zeros(1,10);
for k=1:10
    [~,~,trainLikelihood2f(k,:),valLikelihood2f(k)]=LR(trainC,valC,k,0.8,0.999,0.5,'c');
end
figure();
semilogx(1:500000,trainLikelihood2f(1,:),'k',1:500000,trainLikelihood2f(2,:),'b',1:500000,trainLikelihood2f(3,:),'r',1:500000,trainLikelihood2f(4,:),'g',1:500000,trainLikelihood2f(5,:),'y',1:500000,trainLikelihood2f(6,:),'m',...
1:500000,trainLikelihood2f(7,:),'c',1:500000,trainLikelihood2f(8,:),'k--',1:500000,trainLikelihood2f(9,:),'b--',1:500000,trainLikelihood2f(10,:),'r--');
legend('k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10');
ylabel('Training log-likelihood');
xlabel('Epochs');
hold off;
[maxValLikelihood2f,maxK2f]=max(valLikelihood2f);
[~,testError2f,~,testLikelihood2f]=LR(trainC,testC,maxK2f,0.8,0.999,0.5,'c');





range=linspace(floor(min(dataC(:,1))),ceil(max(dataC(:,1))),100);
figure();
plot(trainC(:,1),trainC(:,2),'+k');
hold on;
plot(testC(:,1),testC(:,2),'ok');
hold on;

[predictedOut2a]=kNN(trainC,[range' round(rand(100,1))],1,'c');
plot(range,predictedOut2a,'k');
hold on;

[predictedOut2b]=kNN(trainC,[range' round(rand(100,1))],minK2b,'c');
plot(range,predictedOut2b,'b');
hold on;

[predictedOut2c]=kNN(trainC,[range' round(rand(100,1))],minK2c,'p');
plot(range,predictedOut2c,'r');
hold on;

[predictedOut2d]=LR(trainC,[range' round(rand(100,1))],1,0.2,0.999,0.5,'t');
plot(range,predictedOut2d,'g');
hold on;


[predictedOut2f]=LR(trainC,[range' round(rand(100,1))],maxK2f,0.8,0.999,0.5,'c');
plot(range,predictedOut2f,'m');
hold on;


legend('Training Data','Test Data','a','b','c','d','f');

xlabel('Input');
ylabel('Prediction');
ylim([-0.5 1.5]);
hold off;


