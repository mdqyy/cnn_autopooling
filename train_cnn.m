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
%Load the digits into workspace
[I,labels,I_test,labels_test] = readMNIST(-1); 
%%

%Define the structure according to [2]
%The net consists of alternating subsampling and convolutional layers,
%followed by a series of fully-connected layers
%Total number of layers
numLayers = 8; 
%Number of subsampling layers (incl. input layer)
numSLayers = 3; 
%Number of convolutional layers
numCLayers = 3; 
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
boolSorting = 0;
%Create an empty convolutional neural network with defined structure
sinet = cnn(numLayers,numFLayers,numInputs,InputWidth,InputHeight,numOutputs,boolSorting);

%Now define the network parameters


%Due to implementation specifics layers are always in pairs. First must be
%subsampling and last (before fully connected) is convolutional layer.
%That's why here first layer is dummy.
sinet.SLayer{1}.SRate = 1;
sinet.SLayer{1}.WS{1} = ones(size(sinet.SLayer{1}.WS{1}));
sinet.SLayer{1}.BS{1} = zeros(size(sinet.SLayer{1}.BS{1}));
sinet.SLayer{1}.TransfFunc = 'purelin';
%Weights 1
%Biases 1


%Second layer - 6 convolution kernels of 28x28 size with 5x5 receptive field
sinet.CLayer{2}.numKernels = 6;
sinet.CLayer{2}.KernWidth = 5;
sinet.CLayer{2}.KernHeight = 5;
%NB: weights are shared among all units in a kernel (feature map)
%Weights 150 (each unit connected to 25 inputs => 25*6 = 150)
%Biases 6 (each unit is also connected to a bias => 1*6 = 6)
% Stride of 1 is assumed


%Third layer - 6 feature maps of 14x14 size connected to 2x2 regions
%Subsampling rate
sinet.SLayer{3}.SRate = 2;
%Weights 6 (each unit connected to 4 inputs mult by a single weight =>
%           1*6=6)
%Biases 6 (each unit also connected to a bias => 1*6=6)


%Forth layer - 16 kernels of 10x10 size with 5x5 receptive field
sinet.CLayer{4}.numKernels = 16;
sinet.CLayer{4}.KernWidth = 5;
sinet.CLayer{4}.KernHeight = 5;
%Weights 400 (16*25 = 400)
%Biases 6 (16*1 = 16)
% TODO: This is wrong, the CLayer has to have different weights for each of
% the SLayer fmaps that it is connected to, as per LeCun's paper

%Fifth layer - 16 kernels of 5x5 size (2x2 subsampling, non-overlapping)
%Subsampling rate
sinet.SLayer{5}.SRate = 2;
%Weights 6
%Biases 6

%Sixth layer - outputs 120 feature maps of 1x1 size with 5x5 receptive field
sinet.CLayer{6}.numKernels = 120;
sinet.CLayer{6}.KernWidth = 5;
sinet.CLayer{6}.KernHeight = 5;
%Weights 3000 (25*120 = 3000)
%Biases 120 (1*120 = 120)

%Seventh layer - fully connected, 84 neurons
sinet.FLayer{7}.numNeurons = 84;
%Weights 10080 (84*120 = 10080)
%Biases 84 (1*84 = 84)

%Eight layer - fully connected, 10 output neurons
sinet.FLayer{8}.numNeurons = numOutputs;
%Weights 840 (10*84 = 840)
%Biases 10 (10*1 = 10)

%Initialize the network
sinet = init(sinet);

%According to [2] the generalisation is better if there's unsimmetry in
%layers connections. Yann LeCun uses this kind of connection map:
sinet.CLayer{4}.ConMap = ...
[1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1;
 1 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1;
 1 1 1 0 0 0 1 1 1 0 0 1 0 1 1 1;
 0 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1;
 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1; 
 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1; 
]';
%but some papers proposes to randomly generate the connection map. So you
%can try it:
%sinet.CLayer{6}.ConMap = round(rand(size(sinet.CLayer{6}.ConMap))-0.1);

% Initialize the first layer to be just the input (weights 1, biases 0)
%sinet.SLayer{1}.WS{1} = ones(size(sinet.SLayer{1}.WS{1}));
%sinet.SLayer{1}.BS{1} = zeros(size(sinet.SLayer{1}.BS{1}));

%In my impementation output layer is ordinary tansig layer as opposed to
%[1,2], but I plan to implement the radial basis output layer

%sinet.FLayer{8}.TransfFunc = 'radbas';


%%
%Now the final preparations
%Number of epochs
sinet.epochs = 3;
%Mu coefficient for stochastic Levenberg-Markvardt
sinet.mu = 0.001;
%Training coefficient
%sinet.teta =  [50 50 20 20 20 10 10 10 5 5 5 5 1]/100000;
sinet.teta =  0.0005;
%0 - Hessian running estimate is calculated every iteration
%1 - Hessian approximation is recalculated every cnet.Hrecomp iterations
%2 - No Hessian calculations are made. Pure stochastic gradient
sinet.HcalcMode = 0;    
sinet.Hrecalc = 300; %Number of iterations to passs for Hessian recalculation
sinet.HrecalcSamplesNum = 50; %Number of samples for Hessian recalculation

%Teta decrease coefficient
sinet.teta_dec = 0.4;

%Images preprocessing. Resulting images have 0 mean and 1 standard
%deviation, and are padded by 4 pixels on all sides => 32x32
%Go inside the preproc_data for details.
[Ip, labtrn] = preproc_data(I,length(I),labels,0,4);
[I_testp, labtst] = preproc_data(I_test,length(I_test),labels_test,0,4);

% Display the important parameters
fprintf('Convnet convolution layer transfer function: %s\n', sinet.CLayer{2}.TransfFunc);
fprintf('Convnet pooling layer transfer function: %s\n', sinet.SLayer{3}.TransfFunc);
fprintf('Convnet pooling layer subsampling function: %s\n', sinet.SLayer{3}.SFunc);
fprintf('Convnet fully connected layer transfer function: %s\n', sinet.FLayer{7}.TransfFunc);
fprintf('\n');

%Actually training
sinet = train(sinet,Ip,labtrn,I_testp,labtst);


