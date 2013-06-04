%% REPORT.m : output the training and testing error on a desired dataset

clear all; clc;

% Load trained model
load cnet.mat;

% Load dataset
% MNIST
[I,labels,I_test,labels_test] = readMNIST();

% Preprocess the data
[Ip, labtrn] = preproc_data(I,length(I),labels,0,4);
[I_testp, labtst] = preproc_data(I_test,length(I_test),labels_test,0,4);

% Compute training error
mcr_train = calcMCR(sinet,Ip, labels, 1:length(labels));

% Compute test error
mcr_test = calcMCR(sinet,I_testp, labels_test, 1:length(labels_test));

fprintf('=====================\n');
fprintf('Performance on MNIST\n');
fprintf('Training misclassification rate: %.2f\n', mcr_train);
fprintf('Test misclassification rate: %.2f\n', mcr_test);
