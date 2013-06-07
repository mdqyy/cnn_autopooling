function [out, cnet] = sim(cnet,inp)

%SIM simulate convolutional neural network: forward propagation. 
% Given new test input, make a prediction of the output.
%
%  Syntax
%  
%    [out, sinet] = sim(cnet,inp)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    inp - input image (e.g.: 32x32 double matrix)
%   Output:
%    cnet - Convolutional neural network with unchanged weights and biases
%    but with saved layers outputs 
%    out - simulated neural network output (predicted class)
%
%(c) Sirotenko Mikhail, 2009

%Supposed that input image is preprocessed to zero mean and 1 deviation
%Subsampling
%if isfield(cnet.SLayer{1}, 'SFunc')
[cnet.SLayer{1}.SS{1}, cnet.SLayer{1}.OS{1}] = subsample(inp,cnet.SLayer{1}.SRate,cnet.SLayer{1}.SFunc);
cnet.SLayer{1}.YS{1} = cnet.SLayer{1}.SS{1}.*cnet.SLayer{1}.WS{1}+cnet.SLayer{1}.BS{1};
%Transfer (activation,sqashing) function 
cnet.SLayer{1}.XS{1} = feval(cnet.SLayer{1}.TransfFunc,cnet.SLayer{1}.YS{1});

%Main layer loop
for k=2:(cnet.numLayers-cnet.numFLayers) %(First layer is dummy, skip it)
    
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
    %S-layer
    
        if cnet.boolSorting == 1
            prevLayer = cnet.OLayer{k-1};
            %Auto "subsampling" function -- multiply the subsampling
            %regions from the OLayer by the weights of the SLayer instead
            %of using a hardcoded function
            XC = reshape(prevLayer.XO,1,[]); %reshape to 1-D vector
            for l=1:prevLayer.numFMaps
                %Pool using learned parameters WS
                [cnet.SLayer{k}.SS{l}, cnet.SLayer{k}.OS{l}] = subsample(XC{l},cnet.SLayer{k}.SRate,cnet.SLayer{k}.SFunc,cnet.SLayer{k}.WS{l},cnet.SLayer{k}.BS{l});
                cnet.SLayer{k}.YS{l} = cnet.SLayer{k}.SS{l}; %NB: no weights here!
                %Apply transfer function
                cnet.SLayer{k}.XS{l} = feval(cnet.SLayer{k}.TransfFunc,cnet.SLayer{k}.YS{l});
            end
        else
            prevLayer = cnet.CLayer{k-1};
            %Reshape output matrix to 1-D vector
            XC = reshape(prevLayer.XC,1,[]);
            for l=1:prevLayer.numFMaps %For all feature maps from previous layer
                %Pool using learned parameters WS
                [cnet.SLayer{k}.SS{l}, cnet.SLayer{k}.OS{l}] = subsample(XC{l},cnet.SLayer{k}.SRate,cnet.SLayer{k}.SFunc,cnet.SLayer{k}.WS{l},cnet.SLayer{k}.BS{l});
                cnet.SLayer{k}.YS{l} = cnet.SLayer{k}.SS{l}; %NB: no weights here!
                %Apply transfer function
                cnet.SLayer{k}.XS{l} = feval(cnet.SLayer{k}.TransfFunc,cnet.SLayer{k}.YS{l});
                
                %%Subsampling (Weights and biases of SLayer are scalars)
                %[cnet.SLayer{k}.SS{l}, cnet.SLayer{k}.OS{l}] = subsample(XC{l},cnet.SLayer{k}.SRate,cnet.SLayer{k}.SFunc);
                %cnet.SLayer{k}.YS{l} = cnet.SLayer{k}.SS{l}*cnet.SLayer{k}.WS{l}+cnet.SLayer{k}.BS{l} ;    
                %%Apply transfer function
                %cnet.SLayer{k}.XS{l} = feval(cnet.SLayer{k}.TransfFunc,cnet.SLayer{k}.YS{l});
            end
        end
        
    elseif index_cLayer == 1
    %C-layer      
        YC = num2cell(zeros(cnet.CLayer{k}.numKernels,1));
        
            for l=1:cnet.CLayer{k}.numKernels %For all convolutional kernels
                % Only convolve the weights with the feature maps from the 
                % previous layer for which the ConMap indicates a '1'
                for m=find(cnet.CLayer{k}.ConMap(l,:))
                    %Convolute and accumulate
                    YC{l} = YC{l}+fastFilter2(cnet.CLayer{k}.WC{l},cnet.SLayer{k-1}.XS{m},'valid')+cnet.CLayer{k}.BC{l};            
                end
                % The transfer function for C-Layers is linear
                cnet.CLayer{k}.XC{l} = YC{l};
            end
         cnet.CLayer{k}.YC = YC;
         if size(cnet.CLayer{k}.XC,1) < size(cnet.CLayer{k}.XC,2) 
            cnet.CLayer{k}.XC = cnet.CLayer{k}.XC'; % need the outgoing in column form
         end
         
    elseif index_oLayer == 1
    %O-layer
        % Compute the sorted inputs, SO
        for l=1:cnet.OLayer{k}.numFMaps
            [cnet.OLayer{k}.SO{l}, cnet.OLayer{k}.OO{l}] = order(cnet.CLayer{k-1}.XC{l}, cnet.OLayer{k}.SRate, cnet.OLayer{k}.SortFunc);
            % Weights = 1, Bias = 0 (the O-Layer only performs sorting)
            cnet.OLayer{k}.YO{l} = cnet.OLayer{k}.SO{l}; %NB no weights to train
            cnet.OLayer{k}.XO{l} = cnet.OLayer{k}.YO{l}; %NB no activation function
        end
    end
end


%Important assumption is made that after last C-Layer all feature maps are
%become 1x1 size, so the output is not a matrix but a vector
%This should be considered while synthesizing the neural network structure
%Convert the cell array of single values to a vector
if cnet.boolSorting == 1
    X_mat = cell2mat(cnet.SLayer{k}.XS);
    XC = X_mat(:)'; % convert previous layer's outputs to a vector, 1 x N
else
    XC = cell2mat(cnet.CLayer{k}.XC)'; % 1 x N vector
end

for k=(cnet.numLayers-cnet.numFLayers+1):cnet.numLayers
    %If the previous layer was not F-Layer
    if (k == cnet.numLayers-cnet.numFLayers+1)
        cnet.FLayer{k}.Y = XC*cnet.FLayer{k}.W+cnet.FLayer{k}.B;
        cnet.FLayer{k}.X = feval(cnet.FLayer{k}.TransfFunc,cnet.FLayer{k}.Y);
    else % previous layer was F-Layer
        cnet.FLayer{k}.Y = cnet.FLayer{k-1}.X*cnet.FLayer{k}.W+cnet.FLayer{k}.B;
        cnet.FLayer{k}.X = feval(cnet.FLayer{k}.TransfFunc,cnet.FLayer{k}.Y);
    end
end

out = cnet.FLayer{k}.X;
