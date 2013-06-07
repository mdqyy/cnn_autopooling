function [cnet,je] = calcje(cnet,e)
%CALCJE Calculation of gradient via backpropagation 
%
%  Syntax
%  
%    [cnet,je] = calcje(cnet,e)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    e - N x 1 vector of errors (difference between network output and 
%        target class) where N is the number of outputs (classes)
%   Output:
%    cnet - convolutional neural network class object with computed
%    gradients (stored in the layers' dEdW, dEdB, dEdX, dEdY variables)
%    je - gradient (or Jacobian times error)
%
%(c) Sirotenko Mikhail, 2009


%Last layer
k = cnet.numLayers;
%Calculate the performance function derivative
cnet.FLayer{k}.dEdX{1} = feval(cnet.Perf, 'dy' , e, cnet.FLayer{k}.Y, cnet.FLayer{k}.X, cnet.Perf);
%Calculating the transfer function derivative
cnet.FLayer{k}.dXdY{1} = feval(cnet.FLayer{k}.TransfFunc,'dn',cnet.FLayer{k}.Y,cnet.FLayer{k}.X); 
%Calculating dE/dY
cnet.FLayer{k}.dEdY{1} = cnet.FLayer{k}.dXdY{1}.*cnet.FLayer{cnet.numLayers}.dEdX{1};
%Check if the previous layer is convolutional or fully-connected
if(cnet.numFLayers~=1) % previous layer is fully-connected
    outp = cnet.FLayer{cnet.numLayers-1}.X;
else % previous layer is not fully-connected
    if cnet.boolSorting == 1
        outp = cnet.SLayer{cnet.numLayers-1}.XS; 
    else
        outp = cnet.CLayer{cnet.numLayers-1}.XC;        
    end
end
%Calculate gradients for weights and biases
% "kron" = Kroneker tensor product = every possible pair of products
cnet.FLayer{k}.dEdW{1} = kron(cnet.FLayer{k}.dEdY{1},outp)';
cnet.FLayer{k}.dEdB{1} = cnet.FLayer{k}.dEdY{1}'; %biases are 1

%Reshape data into single-column vector
je=cnet.FLayer{k}.dEdW{1};
je=[je;cnet.FLayer{k}.dEdB{1}];

if (cnet.numFLayers>1) %If there are more than 1 fully-connected layers
    for k=cnet.numLayers-1:cnet.numLayers-cnet.numFLayers+1
        %Backpropagate error to outputs of this layer
        cnet.FLayer{k}.dEdX{1} = cnet.FLayer{k+1}.W*cnet.FLayer{k+1}.dEdY{1}';
        %Calculating the transfer function derivative
        cnet.FLayer{k}.dXdY{1} = feval(cnet.FLayer{k}.TransfFunc,'dn',cnet.FLayer{k}.Y,cnet.FLayer{k}.X)';
        %Backpropagate error to transfer function inputs
        cnet.FLayer{k}.dEdY{1} = cnet.FLayer{k}.dXdY{1}.*cnet.FLayer{k}.dEdX{1};
        %Check if the previous layer is fully-connected or not
        if(cnet.numLayers-cnet.numFLayers+1==k)
            if cnet.boolSorting == 1
                outp = cell2mat(cnet.SLayer{k-1}.XS); %pooling 
                outp = outp(:);
            else
                outp = cell2mat(cnet.CLayer{k-1}.XC); %convolutional, Nx1
            end
        else
             outp = cnet.FLayer{k-1}.X; %fully-connected
        end
        %Calculate gradients for weights and biases        
        cnet.FLayer{k}.dEdW{1} = kron(cnet.FLayer{k}.dEdY{1},outp); 
        cnet.FLayer{k}.dEdB{1} = cnet.FLayer{k}.dEdY{1};     
        
        %Reshape data into single-column vector
        je=[je;cnet.FLayer{k}.dEdW{1}];
        je=[je;cnet.FLayer{k}.dEdB{1}];   
        
    end
end

k = cnet.numLayers-cnet.numFLayers;
%Backpropagating the error to the layer right before the F-Layers
if cnet.boolSorting == 1
    
    dEdX_vector = cnet.FLayer{k+1}.W * cnet.FLayer{k+1}.dEdY{1}; % (numFMaps x (width x height)) x 1
    cnet.SLayer{k}.dEdX = mat2cell(...
        reshape(dEdX_vector, cnet.SLayer{k}.numFMaps*cnet.SLayer{k}.FMapHeight, cnet.SLayer{k}.FMapWidth), ...
        ones(1,cnet.SLayer{k}.numFMaps)*cnet.SLayer{k}.FMapHeight, [cnet.SLayer{k}.FMapWidth]);

else
    cnet.CLayer{k}.dEdX = num2cell(cnet.FLayer{k+1}.W*cnet.FLayer{k+1}.dEdY{1});
    for l=1:cnet.CLayer{k}.numFMaps
        cnet.CLayer{k}.dXdY{l,1} = feval(cnet.CLayer{k}.TransfFunc,'dn',cnet.CLayer{k}.YC{l},cnet.CLayer{k}.XC{l});
        cnet.CLayer{k}.dEdY{l,1} = cnet.CLayer{k}.dXdY{l} .* cnet.CLayer{k}.dEdX{l};
    end
    %cnet.CLayer{k}.dEdY = cnet.CLayer{k}.dXdY .* cnet.CLayer{k}.dEdX;
end

for k=(cnet.numLayers-cnet.numFLayers):-1:2 %Exclude first layer from loop (it's a dummy)
    
    index_sLayer = 0;
    index_cLayer = 0;
    index_oLayer = 0;
    if (cnet.boolSorting==0 && rem(k,2)==1) || (cnet.boolSorting==1 && rem(k-1,3)==0)
        index_sLayer = 1;
    elseif (cnet.boolSorting==0 && rem(k,2)==0) || (cnet.boolSorting==1 && rem(k-2,3)==0)
        index_cLayer = 1;
    elseif (cnet.boolSorting==1 && rem(k,3)==0)
        index_oLayer = 1;
    end
    
    if index_sLayer == 1
        %Subsampling layer
        %Initialize dE/dX for accumulating the error for the previous layer
        dEdX = num2cell(zeros(cnet.SLayer{k}.numFMaps,1));
        
        % NB: SLayer{k}.WS are the weights for the auto pooling function,
        % NOT the layer weights, i.e. Y = S, S = X(from OLayer) * WS
        
        for l=1:cnet.SLayer{k}.numFMaps %For all feature maps
            %Calculating the transfer function derivative
            cnet.SLayer{k}.dXdY{l} = feval(cnet.SLayer{k}.TransfFunc,'dn',cnet.SLayer{k}.YS{l},cnet.SLayer{k}.XS{l}); 
            %Backpropagate error to transfer function inputs
            cnet.SLayer{k}.dEdY{l} = cnet.SLayer{k}.dXdY{l}.*cnet.SLayer{k}.dEdX{l};
            
            %Calculate the gradient for weights and biases
            % (SRate*SRate shared weights per feature map, used in auto pooling function)
            % NB: Y = S (no weights after pooling)
            %cnet.SLayer{k}.dEdW{l} = sum(sum(cnet.SLayer{k}.dEdY{l}.*cnet.SLayer{k}.SS{l}));
            if cnet.boolSorting == 1
                inp = cnet.OLayer{k-1}.XO{l};
            else
                inp = cnet.CLayer{k-1}.XC{l};
            end
            cnet.SLayer{k}.dEdW{l} = subsample_grad(cnet.SLayer{k}.dEdY{l},cnet.SLayer{k}.SRate,cnet.SLayer{k}.SFunc,inp);
            cnet.SLayer{k}.dEdB{l}=sum(sum(cnet.SLayer{k}.dEdY{l}));
            
            if(k>1) %Backpropagate the error if this is not the first layer
                % For SFunc max or stochastic, need to pass in indices (SLayer.OS)
                % For SFunc auto, need to pass in weights (SLayer.WS)
               dEdX{l} = back_subsample(cnet.SLayer{k}.dEdY{l},cnet.SLayer{k}.SRate,cnet.SLayer{k}.SFunc,cnet.SLayer{k}.OS{l},cnet.SLayer{k}.WS{l});
            end
        end
        
        if(k>1) %Store the accumulated backpropagated error
            if cnet.boolSorting == 1
                cnet.OLayer{k-1}.dEdX = reshape(dEdX,size(cnet.OLayer{k-1}.XO,1),size(cnet.OLayer{k-1}.XO,2),1);
            else
                cnet.CLayer{k-1}.dEdX = reshape(dEdX,size(cnet.CLayer{k-1}.XC,1),size(cnet.CLayer{k-1}.XC,2),1);
            end
        end
        %Reshape data into single-column vector
         je=[je;cell2mat(cnet.SLayer{k}.dEdW')];
         je=[je;cell2mat(cnet.SLayer{k}.dEdB')];
    
%---------------------------------------------------------------------------------------------    
    elseif index_cLayer == 1
         %Convolutional layer
         
         %Initialize temporary variables for accumulating the errors
         dEdX = num2cell(zeros(1,cnet.SLayer{k-1}.numFMaps));
         dEdW = num2cell(zeros(1,cnet.CLayer{k}.numKernels));
         dEdB = num2cell(zeros(1,cnet.CLayer{k}.numKernels));     
         for l=1:cnet.CLayer{k}.numFMaps
             
                %Conv layer may have any transfer function -> need dXdY
                cnet.CLayer{k}.dXdY{l} = feval(cnet.CLayer{k}.TransfFunc,'dn',cnet.CLayer{k}.YC{l},cnet.CLayer{k}.XC{l});
                cnet.CLayer{k}.dEdY{l} = cnet.CLayer{k}.dXdY{l}.*cnet.CLayer{k}.dEdX{l};
                
                %For all feature maps of prev layer which have connections to
                %this layer
                for m=find(cnet.CLayer{k}.ConMap(l,:)) 
                    %Backpropagate and accumulate the error
                    dEdX{m} = dEdX{m}+...
                           back_conv2(cnet.SLayer{k-1}.XS{m}, cnet.CLayer{k}.dEdY{l},cnet.CLayer{k}.WC{l},'err');
                    %Calculate and accumulate the shared weights gradient
                    dEdW{l} = dEdW{l}+...
                           back_conv2(cnet.SLayer{k-1}.XS{m}, cnet.CLayer{k}.dEdY{l},cnet.CLayer{k}.WC{l},'gx'); 

                    %Calculating the shared biases gradient
                    dEdB{l}=dEdB{l} + sum(sum(cnet.CLayer{k}.dEdY{l})); 

                end
         end
         %Storing everything
         cnet.SLayer{k-1}.dEdX = dEdX;
         cnet.CLayer{k}.dEdW = dEdW;
         cnet.CLayer{k}.dEdB = dEdB;
         %Reshape data into single-column vector
         je=[je;reshape(cell2mat(cnet.CLayer{k}.dEdW),[],1)];
         je=[je;cell2mat(cnet.CLayer{k}.dEdB)'];
         
        
%---------------------------------------------------------------------------------------------  
    elseif index_oLayer == 1
        %Ordering layer
        
        %Given SLayer's dEdY, compute OLayer's derivatives
        %Initialize temporary variables for accumulating the gradients
        dEdX = num2cell(zeros(1,cnet.CLayer{k-1}.numFMaps));
        
        % OLayer has no activation function -> unit derivatives
        cnet.OLayer{k}.dEdY = cnet.OLayer{k}.dEdX;
        
        %Compute for every feature map
        for l=1:cnet.OLayer{k}.numFMaps
            
            % OLayer has no activation function -> unit derivatives
            cnet.OLayer{k}.dXdY{l} = ones(size(cnet.OLayer{k}.XO{l}));
            
            % For previous layer (CLayer). NB: dEdY = dEdSO for OLayer
            dEdX{l} = back_order(cnet.OLayer{k}.dEdY{l},cnet.OLayer{k}.SRate,cnet.OLayer{k}.OO{l});
        end
        
        %Store everything
        cnet.CLayer{k-1}.dEdX = reshape(dEdX,size(cnet.CLayer{k-1}.XC,1),size(cnet.CLayer{k-1}.XC,2),1);
        
        % No update to je, no weights to train in OLayer
    end
end

%fprintf('Finished calcje.m, size of je:\n');
%disp(size(je));
%fprintf('Size of cnet:\n');
%disp(cnn_size(cnet));

end
