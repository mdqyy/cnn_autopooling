function sz = cnn_size(cnet)
%cnn_size Calculate the total number of all trainable parameters
%
%  Syntax
%  
%    sz = cnn_size(cnet)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%   Output:
%    sz - number of all trainable parameters 
%
%(c) Sirotenko Mikhail, 2009


sz = 0;
%Loop through the fully-connected layers (the loop is inclusive)
for k=cnet.numLayers:-1:(cnet.numLayers-cnet.numFLayers+1)
    %fprintf('k=%d, size=%d\n', k, numel(cnet.FLayer{k}.W)+numel(cnet.FLayer{k}.B));
    sz = sz + numel(cnet.FLayer{k}.W)+numel(cnet.FLayer{k}.B);
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
        %fprintf('k=%d, size=%d\n', k, numel(cnet.SLayer{k}.WS)*numel(cnet.SLayer{k}.WS{1})+numel(cnet.SLayer{k}.BS)*numel(cnet.SLayer{k}.BS{1}));
        sz = sz + numel(cnet.SLayer{k}.WS)*numel(cnet.SLayer{k}.WS{1})+numel(cnet.SLayer{k}.BS)*numel(cnet.SLayer{k}.BS{1});
    elseif index_cLayer == 1
        %Convolutional layer
        %fprintf('k=%d, size=%d\n', k, numel(cnet.CLayer{k}.WC)*numel(cnet.CLayer{k}.WC{1})+numel(cnet.CLayer{k}.BC)*numel(cnet.CLayer{k}.BC{1}));
        sz = sz + numel(cnet.CLayer{k}.WC)*numel(cnet.CLayer{k}.WC{1})+numel(cnet.CLayer{k}.BC)*numel(cnet.CLayer{k}.BC{1});
    elseif index_oLayer == 1
        %Ordering layer - no trainable weights, only does reordering
        %fprintf('k=%d, size=%d\n', k, 0);
        %sz = sz + numel(cnet.OLayer{k}.WO)*numel(cnet.OLayer{k}.WO{1})+numel(cnet.OLayer{k}.BO)*numel(cnet.OLayer{k}.BO{1});
    end
end

end

