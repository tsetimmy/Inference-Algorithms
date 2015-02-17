function [predictedOut,trainMSEError,testMSEError,trainClassError,testClassError]=NN(trainData,testData,k,mu,momentum,numEpochs,w,method)

numTrainCases = size(trainData,1);
numTestCases = size(testData,1);

predictedOut=zeros(numTestCases,1);

trainMSEError=zeros(numEpochs,1);
testMSEError=zeros(numEpochs,1);
trainClassError=zeros(numEpochs,1);
testClassError=zeros(numEpochs,1);

h=zeros(k,1);
derh=zeros(k,1);
y=0;
dery=0;
del=zeros(2*k,1);
   

for ep=1:numEpochs
    
    delOld=del;
    del=zeros(2*k,1);
    trainMSEError(ep)=0;
    testMSEError(ep)=0;
    trainClassError(ep)=0;
    testClassError(ep)=0;

    for t=1:numTrainCases
        % Forward
        h=trainData(t,1).*w(1:k);
        if method=='c'
            y=sigmoid(sigmoid(h)'*w(k+1:2*k));
            trainMSEError(ep)=trainMSEError(ep)+(trainData(t,2)*log(y)+(1-trainData(t,2))*log(1-y));            
            trainClassError(ep)=trainClassError(ep)+(single(y>=0.5)-trainData(t,2)).^2;
        else
            y=sigmoid(h)'*w(k+1:2*k);
            trainMSEError(ep)=trainMSEError(ep)+(y-trainData(t,2)).^2;
            if method=='t'
                trainClassError(ep)=trainClassError(ep)+(single(y>=0.5)-trainData(t,2)).^2;
            end
        end
        
        % Backward
        if method=='c'
            dery=2*(y-trainData(t,2)).*sigmoid(y).*(1-sigmoid(y));
        else
            dery=2*(y-trainData(t,2));
        end
        derh=sigmoid(h).*(1-sigmoid(h)).*w(k+1:2*k).*dery;
        
        % Update
        del(1:k)=del(1:k)-derh.*trainData(t,1);
        del(k+1:2*k)=del(k+1:2*k)-dery.*sigmoid(h);

    end    
    
    for t=1:numTestCases
        if method=='c'
            predictedOut(t)=sigmoid(sigmoid(testData(t,1).*w(1:k))'*w(k+1:2*k));
            testMSEError(ep)=testMSEError(ep)+(testData(t,2)*log(predictedOut(t))+(1-testData(t,2))*log(1-predictedOut(t)));
            predictedOut(t)=single(predictedOut(t)>=0.5);
            testClassError(ep)=testClassError(ep)+(predictedOut(t)-testData(t,2)).^2;
        else
            predictedOut(t)=sigmoid(testData(t,1).*w(1:k))'*w(k+1:2*k);
            testMSEError(ep)=testMSEError(ep)+(predictedOut(t)-testData(t,2)).^2;
            if method=='t'
                predictedOut(t)=single(predictedOut(t)>=0.5);
                testClassError(ep)=testClassError(ep)+(predictedOut(t)-testData(t,2)).^2;
            end
        end
    end
    
    del=momentum.*delOld + mu.*del./numTrainCases;
    w=w+del;
    trainMSEError(ep)=trainMSEError(ep)/numTrainCases;
    testMSEError(ep)=testMSEError(ep)/numTestCases;
    trainClassError(ep)=trainClassError(ep)/numTrainCases;
    testClassError(ep)=testClassError(ep)/numTestCases;
    if method=='c'
        trainMSEError(ep)=trainMSEError(ep).*numTrainCases;
        testMSEError(ep)=testMSEError(ep).*numTestCases;
    end    
end

end

function [y]=sigmoid(x)
y=1./(1+exp(-x));
end