function [toPlot] = visualizeNet(cnet,plotTitle)
% Given a trained convolutional neural network, cnet, plot all hidden layer
% weights to visualize what is being learnt.
% 
% INPUTS
% cnet - a trained conv net
% 
% OUTPUTS
% none (only figures)
%


    % Determine the number of plots (one row for each layer, 2 plots across)
    numRows = cnet.numLayers;
    numCols = 0;% determined dynamically by adding up num plots for each layer 
    index = 1;  % index of the subplot in the figure
    k = 1;      % index of the next layer to plot

    toPlot = cell(1,1);

    % Collect all data to be plotted - start with input
    fmaps = cnet.SLayer{k}.numFMaps;
    numCols = updateCols(numCols,fmaps);
    for m=1:fmaps
        toPlot{index} = cnet.SLayer{k}.XS{m};
        index = index + 1;
    end

    % Collect all data from the C-O-S or C-S layers
    for k=2:cnet.numLayers-cnet.numFLayers
        nextWeights = cnnGetField(cnet,k,'W');
        fmaps = cnnGetField(cnet,k,'numFMaps');
        numCols = updateCols(numCols,fmaps);
        for m=1:fmaps
            toPlot{index} = nextWeights{m};
            index = index + 1;
        end
    end

    % Collect all data from the F-Layers
    for k=cnet.numLayers-cnet.numFLayers+1:cnet.numLayers
        % FLayer
        nextWeights = cnnGetField(cnet,k,'W');
        toPlot{index} = nextWeights;
        index = index + 1;
    end

    % Now actually plot
    figure;
    colormap(gray);
    for k=1:index-1
        subplot(numRows,numCols,k);
        imagesc(toPlot{k});
        title(['Layer ', num2str(k), ' Weights']);
        colorbar('location','eastoutside');
    end
    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],...
        'Box','off','Visible','off','Units','normalized','clipping','off');
    text(0.5, 1, plotTitle, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top');
    
end

function [cols] = updateCols(cols, newCols)
    if newCols > cols
        cols = newCols;
    end
end