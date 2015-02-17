
close all;


dataR(:,1)=textread('RegressionX.txt','','delimiter',',');
dataR(:,2)=textread('RegressionY.txt','','delimiter',',');
trainR=dataR(1:50,:);
valR=dataR(51:100,:);
testR=dataR(101:200,:);




[~,testError1a]=kNN(trainR,testR,1,'r');


valError1b=zeros(1,10);
for k=1:10
    [~,valError1b(k)]=kNN(trainR,valR,k,'r');
end
[minValError1b,minK1b]=min(valError1b);
[~,testError1b]=kNN(trainR,testR,minK1b,'r');


[~,testError1c]=LR(trainR,testR,1,0.2,0.999,0.5,'r');


valError1d=zeros(1,10);
for k=1:10
    [~,valError1d(k)]=LR(trainR,valR,k,0.2,0.999,0.5,'r');
end
[minValError1d,minK1d]=min(valError1d);
[~,testError1d]=LR(trainR,testR,minK1d,0.2,0.999,0.5,'r');





range=linspace(floor(min(dataR(:,1))),ceil(max(dataR(:,1))),100);
figure();
plot(trainR(:,1),trainR(:,2),'+k');
hold on;
plot(testR(:,1),testR(:,2),'ok');
hold on;
[predictedOut1a]=kNN(trainR,[range' rand(100,1)],1,'r');
plot(range,predictedOut1a,'k');
hold on;
[predictedOut1b]=kNN(trainR,[range' rand(100,1)],minK1b,'r');
plot(range,predictedOut1b,'r');
hold on;
[predictedOut1c]=LR(trainR,[range' rand(100,1)],1,0.2,0.999,0.5,'r');
plot(range,predictedOut1c,'b');
hold on;
[predictedOut1d]=LR(trainR,[range' rand(100,1)],minK1d,0.2,0.999,0.5,'r');
plot(range,predictedOut1d,'g');

legend('Training Data','Test Data','a','b','c','d');
xlabel('Input');
ylabel('Prediction');
hold off;

