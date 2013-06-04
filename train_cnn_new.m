%Convolutional neural network for handwriten digits recognition: training
%and simulation.
%(c)Mikhail Sirotenko, 2009.
%This program implements the convolutional neural network for MNIST handwriten 
%digits recognition, created by Yann LeCun. CNN class allows to make your
%own convolutional neural net, defining arbitrary structure and parameters.
%It is assumed that MNIST database is located in './MNIST' directory.
%References:
%1. Y. LeCun, L. Bottou, G. Orr and K. Muller: Efficient BackProp, in Orr, G.
%and Muller K. (Eds), Neural Networks: Tricks of the trade, Springer, 1998
%URL:http://yann.lecun.com/exdb/publis/index.html
%2. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning
%Applied to Document Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998
%URL:http://yann.lecun.com/exdb/publis/index.html
%3. Patrice Y. Simard, Dave Steinkraus, John C. Platt: Best Practices for
%Convolutional Neural Networks Applied to Visual Document Analysis
%URL:http://research.microsoft.com/apps/pubs/?id=68920
%4. Thanks to Mike O'Neill for his great article, wich is summarize and
%generalize all the information in 1-3 for better understandig for
%programming:
%URL: http://www.codeproject.com/KB/library/NeuralNetRecognition.aspx
%5. Also thanks to Jake Bouvrie for his "Notes on Convolutional Neural
%Networks", particulary for the idea to debug the neural network using
%finite differences
%URL: http://web.mit.edu/jvb/www/cv.html

clear;
clc;
clear classes;
%Load the digits into workspace
[I,labels,I_test,labels_test] = readMNIST(-1); 

%% Architecture
%Total number of layers
numLayers = 6;  
%Number of fully-connected layers
numFLayers = 2;
%Number of input images (simultaneously processed). Need for future
%releases, now only 1 is possible
numInputs = 1; 
%Image width
InputWidth = 32; 
%Image height
InputHeight = 32;
%Number of outputs
numOutputs = 10;
%Whether to use ordered pooling layers (1) or not (0)
boolSorting = 1;
%Create an empty convolutional neural network with defined structure
sinet = cnn(numLayers,numFLayers,numInputs,InputWidth,InputHeight,numOutputs,boolSorting);

%Now define the network parameters


%SLayer{1} - Due to implementation specifics layers are always in pairs. First must be
%subsampling and last (before fully connected) is convolutional layer.
%That's why here first layer is dummy.
sinet.SLayer{1}.SRate = 1;
sinet.SLayer{1}.WS{1} = ones(size(sinet.SLayer{1}.WS{1}));
sinet.SLayer{1}.BS{1} = zeros(size(sinet.SLayer{1}.BS{1}));
sinet.SLayer{1}.TransfFunc = 'purelin';
%Weights 1
%Biases 1


%CLayer{2} - Second layer - 1 convolution kernels of 28x28 size with 5x5 receptive field
sinet.CLayer{2}.numKernels = 1;
sinet.CLayer{2}.KernWidth = 5;
sinet.CLayer{2}.KernHeight = 5;
%NB: weights are shared among all units in a kernel (feature map)
%Weights 25 (each unit connected to 25 inputs => 25*1 = 25)
%Biases 1 (each unit is also connected to a bias => 1*1 = 1)
% Stride of 1 is assumed


%OLayer{3} - Third layer - 1 feature maps of 14x14 size connected to 2x2 regions
%Subsampling rate
sinet.OLayer{3}.SortFunc = 'descend';
sinet.OLayer{3}.SRate = 2;
%Weights 1 (each unit connected to 4 inputs mult by a single weight => 1*1=1)
%Biases 1 (each unit also connected to a bias => 1*1=1)


%SLayer{4} - Fourth layer - 1 feature map of 27x27 suze connected to 2x2 regions
%Subsampling rate
sinet.SLayer{4}.SRate = 2;
%Weights 4 (1*2x2 = 4)
%Biases 1 (1*1 = 1)

%FLayer{5} - Fifth layer - fully connected, 120 neurons
sinet.FLayer{5}.numNeurons = 120;
%Weights 87480 (120x27x27 = 87480)
%Biases 120 (120x1 = 120)

%FLayer{6} - Sixth layer - fully connected, 10 output neurons
sinet.FLayer{6}.numNeurons = numOutputs;
%Weights 1200 (10*120 = 1200)
%Biases 10 (10*1 = 10)

% TOTAL SIZE OF NET: 26 + 2 + 5 + 87600 + 1210 = 88843 trainable weights

%Initialize the network
sinet = init(sinet);


%%
%Now the final preparations
%Number of epochs
sinet.epochs = 1;
%Mu coefficient for stochastic Levenberg-Markquardt
sinet.mu = 0.001;
%Training coefficient
%sinet.teta =  [50 50 20 20 20 10 10 10 5 5 5 5 1]/100000;
sinet.teta =  0.0005;
%0 - Hessian running estimate is calculated every iteration
%1 - Hessian approximation is recalculated every cnet.Hrecomp iterations
%2 - No Hessian calculations are made. Pure stochastic gradient descent.
sinet.HcalcMode = 2;    
sinet.Hrecalc = 300; %Number of iterations to passs for Hessian recalculation
sinet.HrecalcSamplesNum = 50; %Number of samples for Hessian recalculation

%Teta decrease coefficient
sinet.teta_dec = 0.0;

%Images preprocessing. Resulting images have 0 mean and 1 standard
%deviation, and are padded by 4 pixels on all sides => 32x32
%Go inside the preproc_data for details.
[Ip, labtrn] = preproc_data(I,length(I),labels,0,4);
[I_testp, labtst] = preproc_data(I_test,length(I_test),labels_test,0,4);

% Display the important parameters
fprintf('Convnet convolution layer transfer function: %s\n', sinet.CLayer{2}.TransfFunc);
fprintf('Convnet pooling layer transfer function: %s\n', sinet.SLayer{4}.TransfFunc);
fprintf('Convnet pooling layer subsampling function: %s\n', sinet.SLayer{4}.SFunc);
fprintf('Convnet fully connected layer transfer function: %s\n', sinet.FLayer{6}.TransfFunc);
fprintf('\n');

%Actually training
sinet = train(sinet,Ip,labtrn,I_testp,labtst);

