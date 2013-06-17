function [input,cnet,out,d,cost] = testgrad(input)
%
% Construct a simple, 1 cnet.CLayer + 1 cnet.SLayer network, and check gradients
% numerically to make sure they are correct for both and find issues
%
% input -> cnet.CLayer -> cnet.SLayer -> output
%
% INPUTS
% input - a M x M input image, where M is the side length in pixels
%

%% Architecture
cnet.CLayer.CWidth = 3;
cnet.CLayer.CHeight = 3;
cnet.CLayer.Func = 'sigmoid';
cnet.SLayer.SRate = 2;
cnet.SLayer.Func = 'sigmoid';
cnet.FLayer.Func = 'sigmoid';
cnet.FLayer.numOutputs = 10;
cnet.learningRate = 0.001;

%% Initialization
cnet.CLayer.WC = ones(cnet.CLayer.CHeight,cnet.CLayer.CWidth);
cnet.CLayer.BC = 0;
cnet.CLayer.LayerWidth = size(input,2) - cnet.CLayer.CWidth + 1;
cnet.CLayer.LayerHeight = size(input,1) - cnet.CLayer.CHeight + 1;

cnet.SLayer.LayerWidth = floor(cnet.CLayer.LayerWidth/cnet.SLayer.SRate);
cnet.SLayer.LayerHeight = floor(cnet.CLayer.LayerHeight/cnet.SLayer.SRate);

cnet.FLayer.W = ones(cnet.FLayer.numOutputs,cnet.SLayer.LayerWidth*cnet.SLayer.LayerHeight);
cnet.FLayer.B = zeros(cnet.FLayer.numOutputs,1);

% Number of trainable parameters
net_size = numel(cnet.FLayer.W) + numel(cnet.FLayer.B) + numel(cnet.CLayer.WC) + numel(cnet.CLayer.BC);

% Display all sizes
fprintf('cnet.CLayer width x height: %d x %d\n', cnet.CLayer.LayerWidth, cnet.CLayer.LayerHeight);
fprintf('cnet.SLayer width x height: %d x %d\n', cnet.SLayer.LayerWidth, cnet.SLayer.LayerHeight);
fprintf('cnet.FLayer weights dims: %d x %d\n', size(cnet.FLayer.W,1), size(cnet.FLayer.W,2));

% Target class
d = zeros(cnet.FLayer.numOutputs,1);
d(1) = 1;

%% Forward prop

fprintf('Input:\n');
disp(input);

fprintf('Press enter to run forward propagation...\n');
pause;

[cnet, out] = testgrad_sim(cnet,input, 1);
fprintf('Output of the network (pred | targets):\n');
disp([out, d]);
cost = testgrad_cost(cnet,d);
fprintf('Cost (MSE): %.10f\n', cost);


%% Back prop

fprintf('Press enter to run backward propagation...\n');
pause;

grad = nan(net_size,1);
ind = 1;

% Gradients of error wrt cnet.FLayer
cnet.FLayer.dEdX = 1/cnet.FLayer.numOutputs * 2 * (out - d);
cnet.FLayer.dXdY = feval(cnet.FLayer.Func,'dn',cnet.FLayer.Y,cnet.FLayer.X);
cnet.FLayer.dEdY = cnet.FLayer.dEdX .* cnet.FLayer.dXdY;
cnet.FLayer.dEdW = cnet.FLayer.dEdY * cnet.SLayer.XS';
cnet.FLayer.dEdB = cnet.FLayer.dEdY;
sz = numel(cnet.FLayer.dEdW) + numel(cnet.FLayer.dEdB);
grad(ind:ind+sz-1,1) = [cnet.FLayer.dEdW(:); cnet.FLayer.dEdB(:)];
ind = ind + sz;
fprintf('cnet.FLayer dEdX:\n');
disp(cnet.FLayer.dEdX);
fprintf('cnet.FLayer dXdY:\n');
disp(cnet.FLayer.dXdY);
fprintf('cnet.FLayer dEdY:\n');
disp(cnet.FLayer.dEdY);
fprintf('cnet.FLayer dEdW:\n');
disp(cnet.FLayer.dEdW);
fprintf('cnet.FLayer dEdB:\n');
disp(cnet.FLayer.dEdB);

% Gradients of error wrt cnet.SLayer
cnet.SLayer.dEdX = sum(cnet.FLayer.dEdY .* cnet.FLayer.W);
cnet.SLayer.dXdY = feval(cnet.SLayer.Func,'dn',cnet.SLayer.YS,cnet.SLayer.XS);
cnet.SLayer.dEdY = cnet.SLayer.dEdX .* cnet.SLayer.dXdY;
fprintf('cnet.SLayer dEdX:\n');
disp(cnet.SLayer.dEdX);
fprintf('cnet.SLayer dXdY:\n');
disp(cnet.SLayer.dXdY);
fprintf('cnet.SLayer dEdY:\n');
disp(cnet.SLayer.dEdY);


% Gradients of error wrt cnet.CLayer
cnet.CLayer.dEdX = ones(cnet.CLayer.LayerHeight,cnet.CLayer.LayerWidth) .* (cnet.SLayer.dEdY / (cnet.SLayer.SRate * cnet.SLayer.SRate)) ;
cnet.CLayer.dXdY = feval(cnet.CLayer.Func,'dn',cnet.CLayer.YC,cnet.CLayer.XC);
cnet.CLayer.dEdY = cnet.CLayer.dEdX .* cnet.CLayer.dXdY;
cnet.CLayer.dEdW = back_conv2(input, cnet.CLayer.dEdY, cnet.CLayer.WC,'gx');
cnet.CLayer.dEdB = sum(cnet.CLayer.dEdY(:));
sz = numel(cnet.CLayer.dEdW) + numel(cnet.CLayer.dEdB);
grad(ind:ind+sz-1,1) = [cnet.CLayer.dEdW(:); cnet.CLayer.dEdB(:)];
ind = ind + sz;
fprintf('cnet.CLayer dEdX:\n');
disp(cnet.CLayer.dEdX);
fprintf('cnet.CLayer dXdY:\n');
disp(cnet.CLayer.dXdY);
fprintf('cnet.CLayer dEdY:\n');
disp(cnet.CLayer.dEdY);
fprintf('cnet.CLayer dEdW:\n');
disp(cnet.CLayer.dEdW);
fprintf('cnet.CLayer dEdB:\n');
disp(cnet.CLayer.dEdB);

% Update the weights
%cnet = testgrad_updateWeights(grad);


%% Numerically check the gradients

fprintf('Press enter to run checkgrad...\n');
pause;

fprintf('Size of net: %d\n', net_size);
fprintf('Size of computed gradient:\n');
disp(size(grad));

% Perturb the weights a bit
eps = 10^-4;
threshold = 10^-4;
numgrad = nan(net_size,1);
perturb = zeros(net_size,1);
weights = testgrad_weights(cnet);

for m=1:net_size
    perturb(m) = eps;
    %fprintf('Perturbing weight %d\n', m);
    %fprintf('Weight before adaptation: %.10g\n', weights(m));

    % Simulate the net with the perturbed weights in either direction
    [simNetPlus] = testgrad_updateWeights(cnet,-perturb);
    [simNetMinus] = testgrad_updateWeights(cnet,perturb);
    weightsPlus = testgrad_weights(simNetPlus);
    weightsMinus = testgrad_weights(simNetMinus);

    %fprintf('Weight after adaptation (plus): %.10g\n', weightsPlus(m));
    %fprintf('Weight after adaptation (minus): %.10g\n', weightsMinus(m));

    % Compute error for each perturbed net
    [simNetPlus,~] = testgrad_sim(simNetPlus, input, 0);
    [simNetMinus,~] = testgrad_sim(simNetMinus, input, 0);
    if m <= 20
        i = mod(m, 10);
        if i == 0 i = 10; end
        %fprintf('Output of positive net: %.20g\n', simNetPlus.FLayer.X(i));
        %fprintf('Output of negative net: %.20g\n', simNetMinus.FLayer.X(i));
    end

    ePlus = testgrad_cost(simNetPlus,d);
    eMinus = testgrad_cost(simNetMinus,d);
    %fprintf('Difference between errors: %.20g\n', ePlus-eMinus);
    numgrad(m) = (ePlus - eMinus) / (2*eps*cnet.learningRate);
    perturb(m) = 0;

    % Compute difference
    diff = norm(numgrad(m)-grad(m))/norm(numgrad(m)+grad(m));
    if diff > threshold
        fprintf('numgrad diff is BAD for weight %d: %.5f\n', m, diff);
        disp([numgrad(m), grad(m)]);
    else
        fprintf('numgrad diff is GOOD for weight %d: %.20g\n', m, diff);
    end

    %fprintf('\n\n');
end




end
