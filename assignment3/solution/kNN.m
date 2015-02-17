function [predictedOut,error,confusionMat,likelihood]=kNN(trainData,testData,k,method)

numTestCases = size(testData,1);

predictedOut=zeros(numTestCases,1);
error=0;
likelihood=0;
confusionMat=zeros(2);

switch method
    case 'r'
        for t=1:numTestCases
           idx = knnsearch(trainData(:,1),testData(t,1),'k',k);
           predictedOut(t)=mean(trainData(idx',2));
           error=error+(testData(t,2)-predictedOut(t))^2;
        end
        error=error/numTestCases;
    case 'c'
        for t=1:numTestCases
           idx = knnsearch(trainData(:,1),testData(t,1),'k',k);
           predictedOut(t)=mode(trainData(idx',2));
           confusionMat(testData(t,2)+1,predictedOut(t)+1)=confusionMat(testData(t,2)+1,predictedOut(t)+1)+1;
        end
        error=(confusionMat(1,2)+confusionMat(2,1))/numTestCases;               
    case 'p'
        for t=1:numTestCases
           idx = knnsearch(trainData(:,1),testData(t,1),'k',k);
           predictedOut(t)=(length(find(trainData(idx',2)==1))+0.1)/(k+0.2);
           confusionMat(testData(t,2)+1,round(predictedOut(t))+1)=confusionMat(testData(t,2)+1,round(predictedOut(t))+1)+1;
           likelihood=likelihood+(testData(t,2)*log(predictedOut(t))+(1-testData(t,2))*log(1-predictedOut(t)));
        end
        error=(confusionMat(1,2)+confusionMat(2,1))/numTestCases;
end