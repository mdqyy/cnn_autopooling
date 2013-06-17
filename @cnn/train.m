function [cnet] = train(cnet,Ip,labels,I_testp, labels_test)

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

%Initial MCR calculation
mcr(1)=calcMCR(cnet,I_testp, labels_test, 1:length(labels_test));
plot(h_MCRaxes,mcr);
SetText(h_MCRedit,mcr(end));

% By default, Hessian mode is 0
if(cnet.HcalcMode == 1)
    for i=1:cnet.HrecalcSamplesNum
        %Setting the right output to 1, others to 0
        d = zeros(1,10);
        d(labels(i)+1) = 1;
        %Simulating
        [out, cnet] = sim(cnet,Ip{i});    
        %Calculate the error
        e = out-d;
        %Calculate Jacobian times error, or in other words calculate
        %gradient
        [cnet,je] = calcje(cnet,e); 
        [cnet,hx] = calchx(cnet);         
        jj = jj+diag(sparse(hx));
        SetHessianProgress(h_HessPatch,h_HessEdit,i/cnet.HrecalcSamplesNum);
    end
    %Averaging
    jj = jj/cnet.HrecalcSamplesNum;
end

%Initialize performance variables
mcr = nan(cnet.epochs,1);
mcr_test = nan(cnet.epochs,1);
perf_plot = nan(cnet.epochs,1);
perf_plot_test = nan(cnet.epochs,1);

%For all epochs
for t=1:cnet.epochs
    SetText(h_EpEdit,t);
    SetTextHP(h_TetaEdit,cnet.teta);
    %fprintf('Epoch: %d, Learning rate: %.6f\n', t, cnet.teta);
    
    perf = 0.0;
    
    %For all patterns
    for n=1:numPats
        % d=target output. Set the correct target to 1, others to 0.
        d = zeros(1,10);
        d(labels(n)+1) = 1;
        %Simulating
        [out, cnet] = sim(cnet,Ip{n});    
        %Calculate the error
        e = out-d;
        %Calculate Jacobian times error, or in other words calculate
        %gradient
        [cnet,je] = calcje(cnet,e);
        
        if cnet.checkgrad == 1
            %Check the gradients numerically
            [diff,numgrad,grad] = checkgrad(cnet,net_size,Ip{n},d,1,length(cnet.checkgrad_num),cnet.checkgrad_num,je);
            threshold = 10^-4;
            failed_checkgrad = find(diff > threshold, 1);
            if ~isempty(failed_checkgrad)
                fprintf('FAILED checkgrad for the following weights:\n');
                disp(cnet.checkgrad_num(failed_checkgrad));
                fprintf('Approximated grad | Computed grad \n');
                disp([numgrad(failed_checkgrad), grad(failed_checkgrad)]);
            else
                fprintf('PASSED checkgrad for all weights\n');
            end
        end
        
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
                    d = zeros(1,10);
                    d(labels(i)+1) = 1;
                    %Simulating
                    [out, cnet] = sim(cnet,Ip{i});    
                    %Calculate the error
                    e = out-d;
                    %Calculate Jacobian times error, or in other words calculate
                    %gradient
                    [cnet,je] = calcje(cnet,e); 
                    [cnet,hx] = calchx(cnet);         
                    jj = jj+diag(sparse(hx));
                    
                    SetHessianProgress(h_HessPatch,h_HessEdit,(i-stInd)/cnet.HrecalcSamplesNum);
                end
                %Averaging
                jj = jj/cnet.HrecalcSamplesNum;
            end
        end

        
        %The following is usefull for debugging. 
%===========DEBUG
%        tmp(1)=check_finit_dif(cnet,1,Ip{n},d,1);
%===========DEBUG

        perf = perf + e*e'; %Add up squared error for each training example
        %fprintf('Adding %.5f to MSE accumulator\n', e*e');
        %disp([out',d',e']);
        
        if(cnet.HcalcMode == 2) %Gradient descent
            dW = cnet.teta*je;  %Teta is the learning rate
        else
            %Levenberg-Marquardt (combination of GD and Gauss-Newton)
            dW = (jj+cnet.mu*ii)\(cnet.teta*je);    
        end
        
        %Apply calculated weight updates (after every training datum)
        cnet = adapt_dw(cnet,dW);
        
        SetTrainingProgress(h_TrainPatch,h_TrainEdit,(n+(t-1)*numPats)/(numPats*cnet.epochs));
        SetText(h_ItEdit,n);
        drawnow;
        if(~isempty(get(h_AbortButton,'UserData')))
            fprintf('Training aborted \n');
            return;
        end
    end
    
    %Plot performance after every epoch (1 pass through entire dataset)
      
      %Plot training set misclassification rate
      [mcr(t), perf_plot(t), ~] = calcMCR(cnet,Ip, labels, 1:length(labels));
      [mcr_test(t), perf_plot_test(t), ~] = calcMCR(cnet,I_testp, labels_test, 1:length(labels_test));
      plot(h_MCRaxes,1:length(mcr),mcr,'b-',1:length(mcr_test),mcr_test,'r--');
      SetText(h_MCRedit,mcr(t));
      
      %Plot the training set MSE (blue) and test set MSE (red)
      %perf_plot(t) = perf / numPats;
      plot(h_RMSEaxes,1:length(perf_plot),perf_plot,'b-',1:length(perf_plot_test),perf_plot_test,'r--');
      SetText(h_RMSEedit,perf_plot(t));
      
      % Compute test set MSE and test set MCR
      fprintf('Epoch %d, Train MCR: %.2f, Train MSE: %.5f, Test MCR: %.2f, Test MSE: %.5f\n', ...
                t, mcr(t),perf_plot(t),mcr_test(t),perf_plot_test(t));
            
      % Plot what the weights of the hidden layers are learning
      %weightMat = cnet.CLayer{2}.WC;
      %visualize(weightMat,'Weights of CLayer{2}');
            
    %Update learning rate
    cnet.teta = cnet.teta*cnet.teta_dec;
end

%Add legends to the plots
legend(h_RMSEaxes,'Train','Test');
legend(h_MCRaxes,'Train','Test');

%Stop the running clock to see elapsed time
toc

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