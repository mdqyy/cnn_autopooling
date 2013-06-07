 function out = back_order(e, ratio, indices)
 %
 % Given the dEdS for the OLayer, compute dEdX for the previous CLayer.
 % 
 % INPUTS
 % e - dEdS for the OLayer, N x N double matrix
 % ratio - the size of the regions that were ordered, 1 x 1 scalar int
 % indices - the indices of the elements from e in the original input matrix,
 % N x N int matrix
 %
 % OUTPUTS
 % out - dEdX for the CLayer preceding the current OLayer,
 % (N/2+ratio-1)x(N/2+ratio-1) double matrix
 %

     out = nan(size(e,1)/2+ratio-1, size(e,2)/2+ratio-1);

     for k=1:size(out,1)*size(out,2)
         correspondingVal = e(indices==k);
         out(k) = correspondingVal(1);
     end
 
 end