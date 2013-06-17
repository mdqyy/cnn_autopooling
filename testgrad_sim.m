function [simNet, out] = testgrad_sim(cnet,input,verbose)
% Simulate net with given input (i.e. forward prop)

% cnet.CLayer implementation
cnet.CLayer.YC = fastFilter2(cnet.CLayer.WC,input,'valid') + cnet.CLayer.BC;
cnet.CLayer.XC = feval(cnet.CLayer.Func,cnet.CLayer.YC);

% cnet.SLayer implementation
cnet.SLayer.YS = subsample(cnet.CLayer.XC,cnet.SLayer.SRate,'average');
cnet.SLayer.XS = feval(cnet.SLayer.Func,cnet.SLayer.YS);

% cnet.FLayer implementation
cnet.FLayer.Y = cnet.FLayer.W  * cnet.SLayer.XS + cnet.FLayer.B; 
cnet.FLayer.X = feval(cnet.FLayer.Func,cnet.FLayer.Y);

if verbose == 1
    fprintf('cnet.CLayer WC:\n');
    disp(cnet.CLayer.WC);
    fprintf('cnet.CLayer BC:\n');
    disp(cnet.CLayer.BC);
    fprintf('cnet.CLayer YC:\n');
    disp(cnet.CLayer.YC);
    fprintf('cnet.CLayer XC:\n');
    disp(cnet.CLayer.XC);
    
    fprintf('cnet.SLayer YS:\n');
    disp(cnet.SLayer.YS);
    fprintf('cnet.SLayer XS:\n');
    disp(cnet.SLayer.XS);
    
    fprintf('cnet.FLayer YS:\n');
    disp(cnet.FLayer.Y);
    fprintf('cnet.FLayer XS:\n');
    disp(cnet.FLayer.X);
end


out = cnet.FLayer.X;
simNet = cnet;

end