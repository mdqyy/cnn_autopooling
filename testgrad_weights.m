function weights = testgrad_weights(cnet)
% INPUTS
% cnet - a convolutional neural net
% 
% OUTPUTS
% weights - M x 1 vector of doubles, representing all trainable parameters
% of the cnet
%
% where M is the size of the cnet (in terms of number of total trainable
% weights)

weights = [cnet.FLayer.W(:); cnet.FLayer.B(:); cnet.CLayer.WC(:); cnet.CLayer.BC(:)];

end