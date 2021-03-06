 function out = back_order(e, ratio, indices, perc)
 %
 % Given the dEdS for the OLayer, compute dEdX for the previous CLayer.
 % 
 % INPUTS
 % e - dEdS for the OLayer, N x N double matrix
 % ratio - the size of the regions that were ordered, 1 x 1 scalar int
 % indices - the indices of the elements from e in the original input matrix,
 % N x N int matrix
 % perc - percentiles used for ordering forward (or NaN if not used)
 %
 % OUTPUTS
 % out - dEdX for the CLayer preceding the current OLayer,
 % (N/2+ratio-1)x(N/2+ratio-1) double matrix
 %

    if isnan(perc)
        sizeBlock = [ratio, ratio];
    else
        sizeBlock = [length(perc), 1];
    end
    out = nan(size(e,1)/sizeBlock(1)+ratio-1, size(e,2)/sizeBlock(2)+ratio-1);
    
     for k=1:size(out,1)*size(out,2)
         correspondingVal = e(indices==k);
         
         % When using percentiles, not all values from the previous layer
         % are used in the OLayer -> some have weights of zero
         if isempty(correspondingVal)
             out(k) = 0;
         else
             % Sum over all units in the OLayer that depend on this unit in
             % the CLayer (weights are 1 because the operation is ordering)
            out(k) = sum(correspondingVal);
         end
     end
 
 end