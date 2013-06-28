function cnet = init(cnet)
%init initialize cnn object. It should be called after definition of all
%essential parameters of network, such as number of layers, convolution
%kernels, subsampling rates etc. The only thing, that should be initialized
%after calling init is connection matrix!
%
%  Syntax
%  
%    cnet = init(cnet)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%   Output:
%    cnet - initialized convolutional neural network ready for train
%
%(c) Sirotenko Mikhail, 2009

% Use same seed every time
rng default;

%First is a dummy layer by default. All its weights are 1, all biases are 0. 
r = cnet.SLayer{1}.SRate;
cnet.SLayer{1}.WS{1} = ones(floor(cnet.InputHeight/r),floor(cnet.InputWidth/r));
cnet.SLayer{1}.BS{1} = zeros(floor(cnet.InputHeight/r),floor(cnet.InputWidth/r));
cnet.SLayer{1}.FMapWidth = floor(cnet.InputWidth/r);
cnet.SLayer{1}.FMapHeight = floor(cnet.InputHeight/r);

%For now only single image input is valid
cnet.SLayer{1}.numFMaps = 1; 
    
%For all layers except output
for k=2:(cnet.numLayers-cnet.numFLayers)
    
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
        else
            prevLayer = cnet.CLayer{k-1};
        end
    
        %Make a shortcuts
        r = cnet.SLayer{k}.SRate;
        fmw = prevLayer.FMapWidth;
        fmh = prevLayer.FMapHeight;
        
        %Store layer properties
        cnet.SLayer{k}.FMapWidth = floor(fmw/r);
        cnet.SLayer{k}.FMapHeight = floor(fmh/r);
        
        for l=1:prevLayer.numFMaps %For all feature maps
            %Initialize all weights as 1 and biases randomly
            if cnet.boolSorting == 1
                %Need 1 weight for each unit in the pooling region (these
                %weights are shared by all pooling units in a feature map)
                cnet.SLayer{k}.WS{l}=ones(r*r,1);
                cnet.SLayer{k}.BS{l}=rand_std(1,1,1);
            else
                %Need a single weight per feature map (shared among all
                %pooling units and for all pooled units)
                cnet.SLayer{k}.WS{l}= ones(r*r,1);
                cnet.SLayer{k}.BS{l} = rand_std(1, 1, 1);
            end
        end
        %Subsampling layer doesn't change the number of feature maps
        cnet.SLayer{k}.numFMaps = prevLayer.numFMaps;
        cnet.SLayer{k}.dEdX=cell(1,cnet.SLayer{k}.numFMaps);
        
    elseif index_cLayer == 1
    %C-Layer  
    
            %Initialize the feature map size
            kw = cnet.CLayer{k}.KernWidth;
            kh = cnet.CLayer{k}.KernHeight;
            fmw = cnet.SLayer{k-1}.FMapWidth;
            fmh = cnet.SLayer{k-1}.FMapHeight;

            %Feature map is cropped by 'valid' convolution *of stride 1*
            cnet.CLayer{k}.FMapWidth = fmw-kw+1;
            cnet.CLayer{k}.FMapHeight = fmh-kh+1; 
            
            for l=1:cnet.CLayer{k}.numKernels %For all convolution kernels
              %Set random weights
              %Neuron in next layer have the number of inputs equal to
              %number of weight times number of kernels
              cnet.CLayer{k}.WC{l} = rand_std(kh, kw, kh*kw*cnet.CLayer{k}.numKernels);
              cnet.CLayer{k}.BC{l} = rand_std(1, 1, kh*kw*cnet.CLayer{k}.numKernels);
             
            end
            
            %Initialize connection map: by default every feature map connected to 
            %every convolution kernel. 
            if (~cnet.CLayer{k}.ConMap)
                cnet.CLayer{k}.ConMap = ones(cnet.CLayer{k}.numKernels, cnet.SLayer{k-1}.numFMaps);
            end
            cnet.CLayer{k}.numFMaps = cnet.CLayer{k}.numKernels;
            cnet.CLayer{k}.dEdX=cell(1,cnet.CLayer{k}.numFMaps);
    elseif index_oLayer == 1
    %O-Layer
            
            %Make a shortcuts
            prevLayer = cnet.CLayer{k-1};
            r = cnet.OLayer{k}.SRate;
            fmw = prevLayer.FMapWidth;
            fmh = prevLayer.FMapHeight;
            
            % Initialize OLayer parameters
            cnet.OLayer{k}.numFMaps = prevLayer.numFMaps;
            
            % Variable size of FMaps for OLayer depending on the type of
            % ordering function (and how much data we keep)
            if strcmp(cnet.OLayer{k}.SortFunc,'descend') == 1 || strcmp(cnet.OLayer{k}.SortFunc,'ascend') == 1
                sizeBlock = [r, r];
            elseif strcmp(cnet.OLayer{k}.SortFunc, 'percentile') == 1
                sizeBlock = [length(cnet.OLayer{k}.SortPerc), 1];
            end
            
            cnet.OLayer{k}.FMapWidth = (fmw-r+1)*sizeBlock(2);
            cnet.OLayer{k}.FMapHeight = (fmh-r+1)*sizeBlock(1);
            
            for l=1:cnet.OLayer{k}.numFMaps
                %Initialize all weights as 1 and biases as 0
                cnet.OLayer{k}.WS{l} = 1;
                cnet.OLayer{k}.BS{l} = 0;
            end
    end
end

%Initializing fully connected layers
%It is supposed that outputs of last C-layer is cell vector with single 
%values 
for k=(cnet.numLayers-cnet.numFLayers+1):cnet.numLayers
    %Check if this layer is next after C-layer
    if (k == cnet.numLayers-cnet.numFLayers+1)
        if cnet.boolSorting == 1
            prevLayer = cnet.SLayer{k-1};
            cnet.FLayer{k}.W = rand_std(prevLayer.FMapWidth*prevLayer.FMapHeight,cnet.FLayer{k}.numNeurons,prevLayer.numFMaps+1);
        else
            prevLayer = cnet.CLayer{k-1};
            cnet.FLayer{k}.W = rand_std(prevLayer.numFMaps,cnet.FLayer{k}.numNeurons,prevLayer.numFMaps+1);
        end
        cnet.FLayer{k}.B = rand_std(1,cnet.FLayer{k}.numNeurons,prevLayer.numFMaps+1);
    else
        cnet.FLayer{k}.W = rand_std(cnet.FLayer{k-1}.numNeurons,cnet.FLayer{k}.numNeurons,cnet.FLayer{k-1}.numNeurons+1);
        cnet.FLayer{k}.B = rand_std(1,cnet.FLayer{k}.numNeurons,cnet.FLayer{k-1}.numNeurons+1);
    end
end
