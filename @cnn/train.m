function [cnet,performance] = train(cnet,Ip,labels,I_testp, labels_test)

%TRAIN train convolutional neural network using stochastic Levenberg-Marquardt  
%
%  Syntax
%  
%    [cnet, perf_plot] = train(cnet,Ip,labels,I_testp, labtst)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    Ip - cell array, containing preprocessed images of handwriten digits
%    labels - cell array of labels, corresponding to images
%    I_testp - cell array, containing preprocessed images of handwriten
%    digits of test set
%    labtst - cell array of labels, corresponding to images of test set
%   Output:
%    cnet - trained convolutional neural network
%    perf_plot - performance data
%
%(c) Sirotenko Mikhail, 2009

%Initialize GUI
h_gui = cnn_gui();
%Progress bars
h_HessPatch = findobj(h_gui,'Tag','HessPatch');
h_HessEdit = findobj(h_gui,'Tag','HessPrEdit');
h_TrainPatch = findobj(h_gui,'Tag','TrainPatch');
h_TrainEdit = findobj(h_gui,'Tag','TrainPrEdit');
%Axes
h_MCRaxes = findobj(h_gui,'Tag','MCRaxes');
h_RMSEaxes = findobj(h_gui,'Tag','RMSEaxes');
%Info textboxes
h_EpEdit = findobj(h_gui,'Tag','EpEdit');
h_ItEdit = findobj(h_gui,'Tag','ItEdit');
h_RMSEedit = findobj(h_gui,'Tag','RMSEedit');
h_MCRedit = findobj(h_gui,'Tag','MCRedit');
h_TetaEdit = findobj(h_gui,'Tag','TetaEdit');
%Buttons
h_AbortButton = findobj(h_gui,'Tag','AbortButton');

tic;    %Fix the start time
%Coefficient, determining the running estimation of diagonal 
%Hessian approximation leak
gamma = 0.1;  

%Number of training patterns
numPats = length(Ip);
%Calculate the size of network (the total number of weights to train)
net_size = cnn_size(cnet);

% net_size by net_size matrix with 1's on the diagonal, 0's elsewhere
ii = sparse(1:net_size,1:net_size,ones(1,net_size));    
jj = sparse(0);

%Initialize MCR accumulators
mcr = zeros(cnet.epochs,1);
mcr_test = zeros(cnet.epochs,1);
mcr(1)=calcMCR(cnet,I_testp, labels_test, 1:length(labels_test));
plot(h_MCRaxes,mcr);
SetText(h_MCRedit,mcr(end));

%Initialize MSE accumulators
mse = zeros(cnet.epochs,1);
mse_test = zeros(cnet.epochs,1);

%Initialize checkgrad accumulators
checkgrad_diffs = nan(cnet.epochs,length(cnet.checkgrad_num));

% By default, Hessian mode is 0
if(cnet.HcalcMode == 1)
    for i=1:cnet.HrecalcSamplesNum
        %Setting the right output to 1, others to 0
        d = zeros(1,10);
        d(labels(i)+1) = 1;
        %Simulating
        [~, cnet] = sim(cnet,Ip(i));    
        %Calculate Jacobian times error, or in other words calculate
        %gradient
        [cnet,~] = calcje(cnet,d); 
        [cnet,hx] = calchx(cnet);         
        jj = jj+diag(sparse(hx));
        SetHessianProgress(h_HessPatch,h_HessEdit,i/cnet.HrecalcSamplesNum);
    end
    %Averaging
    jj = jj/cnet.HrecalcSamplesNum;
end

%For all epochs
for t=1:cnet.epochs
    SetText(h_EpEdit,t);
    SetTextHP(h_TetaEdit,cnet.teta);
    dW = zeros(net_size,1);
    
    % Used to store squared error for every training example
    perf = nan(numPats,1);
    
    %For all patterns
    for n=1:numPats
        % d=target output. Set the correct target to 1, others to 0.
        d = zeros(1,10);
        d(labels(n)+1) = 1;
        %Simulating
        [out, cnet] = sim(cnet,Ip(n));    
        %Calculate the error
        e = out-d;
        perf(n) = sum(e.^2); %Store the error for each training example
        %Calculate Jacobian times error, or in other words calculate
        %gradient
        [cnet,je] = calcje(cnet,d);
        
        %Calculate Hessian diagonal approximation
        if(cnet.HcalcMode == 0)
            [cnet,hx] = calchx(cnet);         
            %Calculate the running estimate of Hessian diagonal approximation
            jj = gamma*diag(sparse(hx))+sparse((1-gamma)*jj);     
        end
        if(cnet.HcalcMode == 1)
            if(mod(t*numPats+n,cnet.Hrecalc)==0) %If it is time to recalculate Hessian
                if(n+cnet.HrecalcSamplesNum>numPats)
                    stInd = numPats-cnet.HrecalcSamplesNum;
                else
                    stInd = n;
                end
                for i=stInd:stInd+cnet.HrecalcSamplesNum
                    %Setting the right output to 1, others to 0
                    d = zeros(1,cnet.numOutputs);
                    d(labels(i)+1) = 1;
                    %Simulating
                    [~, cnet] = sim(cnet,Ip(i));
                    %Calculate Jacobian times error, or in other words calculate
                    %gradient
                    [cnet,je] = calcje(cnet,d); 
                    [cnet,hx] = calchx(cnet);         
                    jj = jj+diag(sparse(hx));
                    
                    SetHessianProgress(h_HessPatch,h_HessEdit,(i-stInd)/cnet.HrecalcSamplesNum);
                end
                %Averaging
                jj = jj/cnet.HrecalcSamplesNum;
            end
        end
        
        if(cnet.HcalcMode == 2) %Stochastic gradient descent
            if cnet.checkgrad == 1
                %Check the gradients numerically
                [diffs,numgrad,grad] = checkgrad(cnet,net_size,Ip(n),labels(n),1,length(cnet.checkgrad_num),cnet.checkgrad_num,je);
                failed_checkgrad = find(diffs > cnet.checkgrad_threshold, 1);
                if ~isempty(failed_checkgrad)
                    fprintf('FAILED checkgrad for epoch %d\n', t);
                    disp(failed_checkgrad);
                    disp(checkgrad_diffs(t));
                else
                    fprintf('PASSED checkgrad for epoch %d, diff: %g\n', t, diffs(1));
                end
            end
            
            %Apply calculated weight updates (after every training datum)
            dW = cnet.teta*je;  %Teta is the learning rate
            cnet = adapt_dw(cnet,dW);
            
        elseif (cnet.HcalcMode == 3) % Batch gradient descent
            dW = dW + je;
        else
            %Levenberg-Marquardt (combination of GD and Gauss-Newton)
            dW = (jj+cnet.mu*ii)\(cnet.teta*je);   
            
            %Apply calculated weight updates (after every training datum)
            cnet = adapt_dw(cnet,dW);
        end
        
        SetTrainingProgress(h_TrainPatch,h_TrainEdit,(n+(t-1)*numPats)/(numPats*cnet.epochs));
        SetText(h_ItEdit,n);
        drawnow;
        if(~isempty(get(h_AbortButton,'UserData')))
            fprintf('Training aborted \n');
            return;
        end
    end % end of loop through training examples
    
    % Batch gradient descent - update gradients in batch, after pass 
    % through entire dataset
    if (cnet.HcalcMode == 3)
        
        % Average the accumulated gradients
        dW = dW / numPats;
        
        if cnet.checkgrad == 1
            %Check the gradients numerically
            [diffs,numgrad,grad] = checkgrad(cnet,net_size,Ip,labels,1,length(cnet.checkgrad_num),cnet.checkgrad_num,dW);
            failed_checkgrad = find(diffs > cnet.checkgrad_threshold, 1);
            if ~isempty(failed_checkgrad)
                fprintf('FAILED checkgrad for epoch %d\n', t);
                disp(failed_checkgrad);
                disp(checkgrad_diffs(t));
                disp([numgrad', grad']);
            else
                fprintf('PASSED checkgrad for epoch %d, diff: %g\n', t, diffs(1));
            end
        end
        cnet = adapt_dw(cnet, cnet.teta*dW);
    end
    
    %Plot training set MCR for every epoch
    [mcr(t),~,~,~] = calcMCR(cnet, Ip, labels, 1:length(labels));
    [mcr_test(t),mse_test(t),~,~] = calcMCR(cnet, I_testp, labels_test, 1:length(labels_test));
    plot(h_MCRaxes,1:t,mcr(1:t),'b-',1:t,mcr_test(1:t),'r--');
    SetText(h_MCRedit,mcr(t));
    
    %Plot training set MSE (blue) and test set MSE (red) for every epoch
    mse(t) = sum(perf) / numPats;
    plot(h_RMSEaxes,1:t,mse(1:t),'b-',1:t,mse_test(1:t),'r--');
    SetText(h_RMSEedit,mse(t));
    delta_mse = 0;
    if t > 1
        delta_mse = mse(t) - mse(t-1);
    end
    
    fprintf('Epoch %d, Train MCR: %.2f, Train MSE: %.10f (%g, %d), Test MCR: %.2f, Test MSE: %.10f\n', ...
            t, mcr(t),mse(t),delta_mse,delta_mse<0,mcr_test(t),mse_test(t));

    %Decay the learning rate
    cnet.teta = cnet.teta*cnet.teta_dec;
end

%Add legends to the plots
legend(h_RMSEaxes,'Train','Test');
legend(h_MCRaxes,'Train','Test');

%Stop the running clock to see elapsed time
elapsed = toc;

% Pack all performance indicators together to be returned
performance.mseTrain = mse;
performance.mseTest = mse_test;
performance.mcrTrain = mcr;
performance.mcrTest = mcr_test;
performance.elapsedTime = elapsed;

%Sets Hessian progress
%hp - handle of patch
%hs - handle of editbox
%pr - value from 0 to 1
function SetHessianProgress(hp,hs,pr)
xpatch = [0 pr*100 pr*100 0];
set(hp,'XData',xpatch);
set(hs,'String',[num2str(pr*100,'%5.2f'),'%']);
drawnow;


%Sets Training progress
%hp - handle of patch
%hs - handle of editbox
%pr - value from 0 to 1
function SetTrainingProgress(hp,hs,pr)
xpatch = [0 pr*100 pr*100 0];
set(hp,'XData',xpatch);
set(hs,'String',[num2str(pr*100,'%5.2f'),'%']);

%Set numeric text in the specified edit box
%hs - handle of textbox
%num - number to convert and set
function SetText(hs,num)
set(hs,'String',num2str(num,'%5.2f'));

%Set numeric text in the specified edit box with high preceition
%hs - handle of textbox
%num - number to convert and set
function SetTextHP(hs,num)
set(hs,'String',num2str(num,'%5.3e'));