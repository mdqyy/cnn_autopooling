function [out] = cnnGetField(cnet, layerNum, fieldName)
%
% INPUTS
% cnet - a conv net
% layerNum - int, a layer number, e.g. 1
% fieldName - string, e.g. 'W' for weights, 'B' for bias, 'numFMaps' for
% number of feature maps, 'L' for layer itself
%
% OUTPUTS
% out - the requested field for the layer
%
    
    theLayer = nan;
    
    if layerNum <= cnet.numLayers-cnet.numFLayers
        if (cnet.boolSorting==0 && rem(layerNum,2)==1) || (cnet.boolSorting==1 && rem(layerNum-1,3)==0)
            % S-Layer
            theLayer = cnet.SLayer{layerNum};
            theLayer.type = 'S';
        elseif (cnet.boolSorting==0 && rem(layerNum,2)==0) || (cnet.boolSorting==1 && rem(layerNum-2,3)==0)
            % C-Layer
            theLayer = cnet.CLayer{layerNum};
            theLayer.type = 'C';
        elseif (cnet.boolSorting==1 && rem(layerNum,3)==0)
            % O-Layer
            theLayer = cnet.OLayer{layerNum};
            theLayer.type = 'O';
        end
    else
       % F-Layer
       theLayer = cnet.FLayer{layerNum};
       theLayer.type = '';
    end
    
    % Resolve what type of information is requested from the cnet
    if strcmp(fieldName, 'L') == 1
        out = theLayer;
    elseif strcmp(fieldName, 'W') == 1
        f = [fieldName, theLayer.type];
        out = theLayer.(f);
    elseif strcmp(fieldName, 'B') == 1
        f = [fieldName, theLayer.type];
        out = theLayer.(f);
    elseif strcmp(fieldName, 'numFMaps') == 1
        out = theLayer.numFMaps;
    end
    

end