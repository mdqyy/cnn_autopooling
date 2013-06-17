function [outNet] = testgrad_updateWeights(cnet,grad)
% INPUTS
% cnet - a convolutional neural net
% grad - M x 1 vector of gradients, to be applied to all the weights
%
% OUTPUTS
% outNet - the same convolutional net with saved updated parameters in all
% layers
%
% where M is the size of the cnet (in terms of number of trainable weights)
% 

ind = 1;

% Update FLayer weights
sz = numel(cnet.FLayer.W);
cnet.FLayer.W = cnet.FLayer.W - reshape(cnet.learningRate * grad(ind:ind+sz-1,1), size(cnet.FLayer.W) );
ind = ind + sz;
sz = numel(cnet.FLayer.B);
cnet.FLayer.B = cnet.FLayer.B - reshape(cnet.learningRate * grad(ind:ind+sz-1,1), size(cnet.FLayer.B) );
ind = ind + sz;

% Update CLayer weights
sz = numel(cnet.CLayer.WC);
cnet.CLayer.WC = cnet.CLayer.WC - reshape(cnet.learningRate * grad(ind:ind+sz-1,1), size(cnet.CLayer.WC) );
ind = ind + sz;
sz = numel(cnet.CLayer.BC);
cnet.CLayer.BC = cnet.CLayer.BC - reshape(cnet.learningRate * grad(ind:ind+sz-1,1), size(cnet.CLayer.BC) );

outNet = cnet;

end