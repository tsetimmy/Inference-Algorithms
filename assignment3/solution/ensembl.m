%function [predictedOut,error]=ensembl(trainData,testData,kVal1b,kVal1c,kVal1e,kVal1f,kVal1g3,muNN1g,wVal1g1,wVal1g2,wVal1g3,numEpochs1g4,kVal1h3,muNN1h,wVal1h1,wVal1h2,wVal1h3,numEpochs1h4,ind)

function [predictedOut,error]=ensembl(trainData,testData,kVal1b,kVal1c,kVal1f,kVal1g3,muNN1g,wVal1g1,wVal1g2,wVal1g3,numEpochs1g4,kVal1h3,muNN1h,wVal1h1,wVal1h2,wVal1h3,numEpochs1h4,ind)



numTestCases = size(testData,1);
mu=0.8;
momentum=0;
thresh=0.5;
numEpochs=100;

numS=size(ind,1);

predictedOut=zeros(numTestCases,2+numS);
error=zeros(2+numS,1);

[predictedOutVec(1,:),errorVec(1)]=kNN(trainData,testData,1,'c');
%display('Done: 1');
[predictedOutVec(2,:),errorVec(2)]=kNN(trainData,testData,kVal1b,'c');
%display('Done: 2');
[predictedOutVec(3,:),errorVec(3)]=kNN(trainData,testData,kVal1c,'p');
predictedOutVec(3,:)=single(predictedOutVec(3,:)>=thresh);
%display('Done: 3');
[predictedOutVec(4,:),errorVec(4)]=LR(trainData,testData,1,mu,momentum,thresh,'t');
%display('Done: 4');
%[predictedOutVec(5,:),errorVec(5)]=LR(trainData,testData,kVal1e,mu,momentum,thresh,'t');
%display('Done: 5');
[predictedOutVec(5,:),errorVec(5)]=LR(trainData,testData,kVal1f,mu,momentum,thresh,'c');
%display('Done: 6');



[predictedOutVec(6,:),~,~,~,tempError]=NN(trainData,testData,5,muNN1g,momentum,numEpochs,0.2*rand(2*5,1)-0.1,'t');
errorVec(6)=tempError(end);
%display('Done: 7');
[predictedOutVec(7,:),~,~,~,tempError]=NN(trainData,testData,5,muNN1g,momentum,numEpochs,wVal1g1,'t');        
errorVec(7)=tempError(end);
%display('Done: 8');
[predictedOutVec(8,:),~,~,~,tempError]=NN(trainData,testData,kVal1g3,muNN1g,momentum,numEpochs,wVal1g2,'t');
errorVec(8)=tempError(end);
%display('Done: 9');
[predictedOutVec(9,:),~,~,~,tempError]=NN(trainData,testData,10,muNN1g,momentum,numEpochs1g4,wVal1g3,'t');
errorVec(9)=tempError(end);
%display('Done: 10');



[predictedOutVec(10,:),~,~,~,tempError]=NN(trainData,testData,5,muNN1h,momentum,numEpochs,0.2*rand(2*5,1)-0.1,'c');
errorVec(10)=tempError(end);
%display('Done: 11');
[predictedOutVec(11,:),~,~,~,tempError]=NN(trainData,testData,5,muNN1h,momentum,numEpochs,wVal1h1,'c');        
errorVec(11)=tempError(end);
%display('Done: 12');
[predictedOutVec(12,:),~,~,~,tempError]=NN(trainData,testData,kVal1h3,muNN1h,momentum,numEpochs,wVal1h2,'c');
errorVec(12)=tempError(end);
%display('Done: 13');
[predictedOutVec(13,:),~,~,~,tempError]=NN(trainData,testData,10,muNN1h,momentum,numEpochs1h4,wVal1h3,'c');
errorVec(13)=tempError(end);
%display('Done: 14');

predictedOut(:,1)=mode(predictedOutVec);
error(1)=sum(abs(predictedOut(:,1) - testData(:,2)))/numTestCases;
for t=1:numTestCases
    finalVec=[];
    %for i=1:14
    for i=1:13
        finalVec=[finalVec predictedOutVec(i,t).*ones(1,numTestCases-errorVec(i).*numTestCases)];
    end
    predictedOut(t,2)=mode(finalVec);
end
error(2)=sum(abs(predictedOut(:,2) - testData(:,2)))/numTestCases;
for i=1:numS   
    predictedOut(:,2+i)=mode(predictedOutVec(ind(i,:)~=0,:));
    error(2+i)=sum(abs(predictedOut(:,2+i) - testData(:,2)))/numTestCases;
end