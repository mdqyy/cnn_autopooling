% visualize  :         plots sqrt(M) x sqrt(M) pixel images of 
%                      handwritten digits
%
% INPUTS
% inputData  :         a 1 x N cell array, where each element is an 
%                      sqrt(M) x sqrt(M) image
% plotTitle  :         string, specifying the title for the plot
%
% where N      is the number of training examples
%       M      is the number of dimensions (e.g. 256 for 16 x 16 images)
%
function [] = visualize(inputData, plotTitle)

    % Reshape the cell array into an M x N matrix
    inputData = cell2mat(arrayfun(@(x) [inputData{1,x}(:)], 1:size(inputData,2), 'un', 0));
    
    % Compute the rows and columns for the image
    [M N] = size(inputData);
    imageWidth = round(sqrt(M));
    imageHeight = M / imageWidth;
    
    % Choose the number of rows and columns to display such that the data
    % appears in a roughly square output window
    displayRows = floor(sqrt(N));
    displayCols = ceil(N / displayRows);
    
    % Padding between images
    pad = 1;
    
    % Initialize the display to be black
    output = - ones(pad + displayRows * (imageHeight + pad), ...
                       pad + displayCols * (imageWidth + pad));
    
    nextImage = 1;
    for row = 1:displayRows
       for col = 1:displayCols
           if nextImage > N
               break;
           end
           
           % Place the next digit onto the output array
           output(pad + (row-1) * (imageHeight + pad) + (1:imageHeight), ...
               pad + (col-1) * (imageWidth + pad) + (1:imageWidth)) = ...
               reshape(inputData(:,nextImage), imageHeight, imageWidth);
           
           nextImage = nextImage + 1;
       end
       if nextImage > N
           break;
       end
    end
    
    % Display the image on a new plot
    figure;
    colormap(gray);
    imagesc(output, [-1 1]);
    axis image off;
    colorbar('location','eastoutside');
    title(plotTitle);
    
    % Force update the image
    drawnow;
end