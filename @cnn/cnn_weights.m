function weights = cnn_weights(cnet)
%cnn_weights Get all trainable parameters of the network as a single-column vector
%
%  Syntax
%  
%    weights = cnn_weights(cnet)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%   Output:
%    weights - N x 1 doubles, a vector of all trainable parameters (in the
%    same order as the gradients returned by calcje, i.e. the first weight
%    is the last layer's first weight, the last weight is the first layer's
%    last bias weight).
% 
% where N is the size of the network (i.e. the total number of trainable
% parameters)
%
%(c) Maria Yancheva, 2013

% Initialize
net_size = cnn_size(cnet);
weights = nan(net_size,1);
ptr = 1;

%Loop through the fully-connected layers (the loop is inclusive)
for k=cnet.numLayers:-1:(cnet.numLayers-cnet.numFLayers+1)
    sz = numel(cnet.FLayer{k}.W)+numel(cnet.FLayer{k}.B);
    weights(ptr:ptr+sz-1) = [cnet.FLayer{k}.W(:); cnet.FLayer{k}.B(:)];
    ptr = ptr + sz;
end

%All other layers
for k=(cnet.numLayers-cnet.numFLayers):-1:2 %first layer is dummy
    
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
        for m=1:cnet.SLayer{k}.numFMaps
            sz = numel(cnet.SLayer{k}.WS{m})+numel(cnet.SLayer{k}.BS{m});
            weights(ptr:ptr+sz-1) = [cnet.SLayer{k}.WS{m}(:); cnet.SLayer{k}.BS{m}(:)];
            ptr = ptr + sz;
        end    
        
    elseif index_cLayer == 1
        %Convolutional layer
        for m=1:cnet.CLayer{k}.numFMaps
            sz = numel(cnet.CLayer{k}.WC{m})+numel(cnet.CLayer{k}.BC{m});
            weights(ptr:ptr+sz-1) = [cnet.CLayer{k}.WC{m}(:); cnet.CLayer{k}.BC{m}(:)];
            ptr = ptr + sz;
        end
        
    elseif index_oLayer == 1
        %Ordering layer - no trainable weights, only does reordering
    end
end

end