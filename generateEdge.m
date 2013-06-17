function [example] = generateEdge(size, thickness)
% Generate square image with a simple diagonal edge (from top right to 
% bottom left) of varying thickness.
% 
% INPUTS
% size - 1 x 1 int, representing the side length of the square image, in
% pixels
% thickness - 1 x 1 int, representing the thickness of the edge, in pixels
% (an edge of thickness N is 2N-1 pixels wide)
% 
% OUTPUTS
% example - size x size double matrix, representing an image with an edge
% of specified thickness
% 

    example = zeros(size,size);
    for k=1:size
        m = size - k + 1;
        startIndex = max(m-thickness+1,1);
        endIndex = min(m+thickness-1,size);
        example(k,startIndex:endIndex) = 1;
    end

end

