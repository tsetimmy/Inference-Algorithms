function [predictedOut,error,trainLikelihood,testLikelihood]=LR(trainData,testData,k,mu,momentum,thresh,method)

numTrainCases = size(trainData,1);
numTestCases = size(testData,1);

w=0.2*rand(k+1,1)-0.1;

augTrainData=zeros(numTrainCases,k+1);
localAug=zeros(k+1,1);
for t=1:numTrainCases
    for n=1:k+1
    localAug(n)=trainData(t,1)^(n-1);    
    end
    augTrainData(t,:)=localAug';
end
normMean=mean(augTrainData(:,2:k+1));
normSigma=sqrt(var(augTrainData(:,2:k+1)));
augTrainData(:,2:k+1)=(augTrainData(:,2:k+1)-repmat(normMean,numTrainCases,1))./repmat(normSigma,numTrainCases,1);

augTestData=zeros(numTestCases,k+1);
localAug=zeros(k+1,1);
for t=1:numTestCases
    for n=1:k+1
    localAug(n)=testData(t,1)^(n-1);    
    end
    augTestData(t,:)=localAug';
end
augTestData(:,2:k+1)=(augTestData(:,2:k+1)-repmat(normMean,numTestCases,1))./repmat(normSigma,numTestCases,1);

if method=='c'
    i=1;
    grad=ones(k+1,1);
    while max(grad) > 1e-10 && i < 500000
        [trainLikelihood(i) grad]=costFn(w,augTrainData,trainData(:,2),'c');
        trainLikelihood(i)=-trainLikelihood(i);
        w = w + mu.*grad;
        i=i+1;
    end
    display('Done!');
    trainLikelihood=[trainLikelihood trainLikelihood(end)*ones(1,500000-length(trainLikelihood))];
else
    options = optimset('GradObj','on','MaxIter',100000,'TolFun',1e-10,'MaxFunEvals',100000);%,'Display','iter');
    [w,~] = fminunc (@(w) (costFn(w,augTrainData,trainData(:,2),'r')),w,options);  
end

if method=='r'
    predictedOut=augTestData*w;
    error=sum((predictedOut - testData(:,2)).^2)/numTestCases;
elseif method=='t'
    predictedOut=single(augTestData*w > thresh);
    error=sum(abs(predictedOut - testData(:,2)))/numTestCases;
else
    predictedOut=sigmoid(augTestData*w);    
    testLikelihood=sum(testData(:,2).*log(predictedOut) + (1-testData(:,2)).*log(1-predictedOut));    
    predictedOut=single(predictedOut>thresh);
    error=sum(abs(predictedOut - testData(:,2)))/numTestCases;    
end

end

function [error,grad]=costFn(w,x,y,mode)
if mode=='c'
    error=-sum(y.*log(sigmoid(x*w)) + (1-y).*log(1-sigmoid(x*w)));
    grad=2*x'*(y - sigmoid(x*w))./size(x,1);
else
    error=sum((x*w - y).^2)/size(x,1);
    grad=2*x'*(x*w - y)./size(x,1);    
end
end

function [y]=sigmoid(x)
y=1./(1+exp(-x));
end