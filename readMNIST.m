function [I,labels,I_test,labels_test] = readMNIST(num)
%readMNIST MNIST handwriten image database reading.
%
%  Syntax
%  
%    [I,labels,I_test,labels_test] = readMNIST(num)
%    
%  Description
%   Input:
%    id - cell array of images 28x28 size
%    num - number of images to process
%    labels - cell array of labels, corresponding to images
%    rand_on - parameter, defining if it is necessary to randomly pick a
%    pairs of image/label
%   Output:
%    I - cell array of training images 28x28 size
%    labels - vector of labels (true digits) for training set
%    I_test - cell array of testing images 28x28 size
%    labels_test - vector of labels (true digits) for testing set
%
%(c) Sirotenko Mikhail, 2009
%===========Loading training set

%Check if we have MNIST dataset 

if num == -1 % read the minimal example (1 train digit=0, 1 test digit=0)
    load minimal.mat;
    
else % read from the MNIST dataset, a specified number of examples

    % if no number of images provided, assume all images need to be read
    if nargin < 1
        num = -1;
    end

    path = '../../Datasets/MNIST/train-images.idx3-ubyte';
    if(~exist(path,'file'))
        error('Training set of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
    end
    fid = fopen(path,'r','b');  %big-endian
    magicNum = fread(fid,1,'int32');    %Magic number
    if(magicNum~=2051) 
        display('Error: cant find magic number');
        return;
    end
    imgNum = fread(fid,1,'int32');  %Number of images
    rowSz = fread(fid,1,'int32');   %Image height
    colSz = fread(fid,1,'int32');   %Image width

    if(num<imgNum && num ~= -1) 
        imgNum=num; 
    end

    for k=1:imgNum
        I{k} = uint8(fread(fid,[rowSz colSz],'uchar'));
    end
    fclose(fid);

    %============Loading labels
    path = '../../Datasets/MNIST/train-labels.idx1-ubyte';
    if(~exist(path,'file'))
        error('Training labels of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
    end
    fid = fopen(path,'r','b');  %big-endian
    magicNum = fread(fid,1,'int32');    %Magic number
    if(magicNum~=2049) 
        display('Error: cant find magic number');
        return;
    end
    itmNum = fread(fid,1,'int32');  %Number of labels

    if(num<itmNum && num ~= -1) 
        itmNum=num; 
    end

    labels = uint8(fread(fid,itmNum,'uint8'));   %Load all labels

    fclose(fid);

    %============All the same for test set
    path = '../../Datasets/MNIST/t10k-images.idx3-ubyte';
    if(~exist(path,'file'))
        error('Test images of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
    end

    fid = fopen(path,'r','b');  
    magicNum = fread(fid,1,'int32');    
    if(magicNum~=2051) 
        display('Error: cant find magic number');
        return;
    end
    imgNum = fread(fid,1,'int32');  
    rowSz = fread(fid,1,'int32');   
    colSz = fread(fid,1,'int32');   

    if(num<imgNum && num ~= -1) 
        imgNum=num; 
    end

    for k=1:imgNum
        I_test{k} = uint8(fread(fid,[rowSz colSz],'uchar'));
    end
    fclose(fid);

    %============Test labels
    path = '../../Datasets/MNIST/t10k-labels.idx1-ubyte';
    if(~exist(path,'file'))
        error('Test labels of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
    end

    fid = fopen(path,'r','b');  
    magicNum = fread(fid,1,'int32');    
    if(magicNum~=2049) 
        display('Error: cant find magic number');
        return;
    end
    itmNum = fread(fid,1,'int32');  
    if(num<itmNum && num ~= -1) 
        itmNum=num; 
    end
    labels_test = uint8(fread(fid,itmNum,'uint8'));   

    fclose(fid);
end