 function out = subsample_grad(e, ratio, operation, x) 
 %
 % Given dEdS, compute dEdW for SLayer
 %
 % INPUTS
 % e - dEdS, N x N double matrix
 % ratio - 1 x 1 int scalar
 % x - the original matrix before subsampling
 % weights - ratio x ratio weights used for the auto pooling function
 %
 % OUTPUTS
 % out - dEdW, ratio*ratio x 1 double matrix (there are ratio*ratio weights
 % only because they are shared by all units in the SLayer)
 %
 % Example:
 % x = [  1     5     9    13
 %        2     6    10    14
 %        3     7    11    15
 %        4     8    12    16]
 % 
 % ratio = 2 
 %
 % operation = 'auto'
 %
 % e = [ 1  1
 %       1  1]
 %
 % out = [ 24  40
 %         28  44]
 %
 
     % Initialization
     out = nan(ratio*ratio,1);
     size_e = size(e);
     size_x = size(x);
     e = e(:);

     % Only need to compute grad if the pooling function is learned
     % automatically with parameters 'weights'
     if strcmp(operation, 'auto') == 1
    
        % Every row of temp represents the indices of 1 ratio x ratio block
        temp = nan(size(e,1)*size(e,2),ratio*ratio);
        for k=1:ratio
            for m=1:ratio
                % Find indices in original matrix that contribute to this entry
                % in "e"
                ix = combinations(m:ratio:size_e(1)*ratio,k:ratio:size_e(2)*ratio,2);
                linind = sub2ind(size_x, ix(:,1), ix(:,2));
                temp(:,(k-1)*ratio+m) = linind;
                out((k-1)*ratio+m,1) = sum(e .* x(linind));
            end
        end
     end

 end