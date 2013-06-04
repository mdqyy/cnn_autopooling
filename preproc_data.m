function [pd,labnew] = preproc_data(id,n,labels,rand_on,padding)
%preproc_data MNIST handwriten image database preprocessing.
% If the input consists of images of size 28x28, then the script returns 
% images of size 32x32, padded with zeros and normalized to mean 0 and 
% standard deviation of 1. Otherwise, the images are simply preprocessed,
% not padded.
%
%
%
%  Syntax
%  
%    [pd,labnew] = preproc_data(id,n,labels,rand_on)
%    
%  Description
%   INPUT:
%    id - cell array of images 28x28 size
%    n - number of images to process
%    labels - cell array of labels, corresponding to images
%    rand_on - parameter, defining if it is necessary to randomly pick a
%    pairs of image/label (if 1 then the returned data is not only
%    preprocessed but also shuffled)
%    padding - the amount of white space (zero) padding to be added 
%    around the image. If zero, then none is added.
%
%   OUTPUT:
%    pd - cell array of processed images with 0 mean, 1 standard
%    deviation and increased size (from 28x28 to 32x32)
%    labnew - cell array of labels, corresponding to that images
%
%(c) Sirotenko Mikhail, 2009

% preallocate for speed
labnew = nan(n,1);
randd = cell(1,n);
pd = cell(1,n);

% The padding has to be even on all sides of the image
if padding > 0 && mod(padding, 2) ~= 0
    padding = padding + 1;
end

% preprocess all the data
for k=1:n

    if(rand_on==1)        
        rand_num = ceil(rand(1,1)*length(id));
    else
        rand_num = k;
    end
    labnew(k) = labels(rand_num);
    
    if padding > 0
        [dim1, dim2] = size(id{rand_num});
        randd{k} = zeros(dim1 + padding, dim2 + padding);
        randd{k}(padding/2+1:dim1+padding/2,padding/2+1:dim2+padding/2)=double(id{rand_num});
    else
        
    end
    %pd{k} = reshape(mapstd(reshape(randd{k},1,[])),32,32);
    gain = 1./ std(randd{k}(:));
    pd{k} = (randd{k} - mean(randd{k}(:))).*gain;
    
end

