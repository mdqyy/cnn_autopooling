 function [out,indices] = subsample(in, ratio, operation, weights, bias) 
 %
 % INPUTS:
 % in - N x N input image (e.g.: 32x32 matrix)
 % ratio - the subsampling rate, e.g. 2 means to halve the image
 % operation - 'average', 'max', 'stochastic', 'auto'
 % weights - optional. ratio x ratio matrix of doubles. Only used with 
 % 'auto' operation where the pooling function is learned rather than 
 % predefined
 % bias - optional. Scalar. Only used with 'auto' operation
 % 
 % 'stochastic' is implemented as defined in the paper:
 % [1] Zeiler and Fergus. Stochastic Pooling for Regularization of Deep
 % Convolutional Neural Networks. NYU. 2013.
 %
 % OUTPUTS:
 % out - N/ratio x N/ratio subsampled image (smaller in size)
 % indices - N/ratio x N/ratio indices of the subsampled image. Only
 % returned for operations 'max' and 'stochastic'.
 % 
 
    % By default, perform average
    if nargin < 3
        operation = 'average';
    end

    indices = nan;

    if ratio == 1
        out = in;
    else
        if strcmp(operation, 'average') == 1

            out = 0;
            for k=1:ratio
                for l=1:ratio
                    out = out+in(1+(k-1):ratio:size(in,1),1+(l-1):ratio:size(in,2));
                end
            end
            out = out/(ratio*ratio);

        elseif strcmp(operation, 'max') == 1
            % each row of 'temp' contains one ratio x ratio block
            % (in row-major order, i.e. row 2 represents block
            % on row 2, col 1 of the original input matrix)
            temp = nan(size(in,1)/ratio * size(in,2)/ratio, ratio*ratio);
            linind = nan(size(in,1)/ratio*size(in,2)/ratio, ratio*ratio);
            for k=1:ratio
                for m=1:ratio
                    temp2 = in(m:ratio:size(in,1), k:ratio:size(in,2));
                    temp(:,(k-1)*ratio+m) = temp2(:);

                    % Compute the linear indices of the elements in
                    % nextRegion
                    combos = combinations(m:ratio:size(in,1),k:ratio:size(in,2),2);
                    linind(:,(k-1)*ratio+m) = sort( sub2ind(size(in), combos(:,1), combos(:,2) )); % col-wise

                end
            end

            % Compute the max for each region
            [outMax, indices] = max(temp, [], 2);
            out = reshape( outMax, size(in,1)/ratio, size(in,2)/ratio );

            % Convert the local "indices" into overall matrix
            % indices (linearly indexing the original "in" matrix)
            indices_lin = sub2ind(size(linind), 1:size(linind,1), indices');
            indices = reshape( linind(indices_lin), size(in,1)/ratio, size(in,2)/ratio );

        elseif strcmp(operation, 'stochastic') == 1

            % each row of temp represents one ratio x ratio block
            temp = nan(size(in,1)/ratio * size(in,2)/ratio,ratio*ratio);
            linind = nan(size(in,1)/ratio*size(in,2)/ratio, ratio*ratio);
            for k=1:ratio
                for m=1:ratio
                    temp2 = in(m:ratio:size(in,1), k:ratio:size(in,2));
                    temp(:,(k-1)*ratio+m) = temp2(:);

                    % Compute the linear indices of the elements in
                    % nextRegion
                    combos = combinations(m:ratio:size(in,1),k:ratio:size(in,2),2);
                    linind(:,(k-1)*ratio+m) = sort( sub2ind(size(in), combos(:,1), combos(:,2) )); % col-wise

                end
            end

            % Assume all inputs are positive or zero! This is
            % handled by using ReLU activation on the conv layers,
            % but just in case handle it here:
            temp(temp < 0) = 0;

            % each column of 'temp' represents one 4x4 block =>
            % convert the values in each block into probabilities
            prob = temp ./ repmat(sum(temp,1), size(temp,1), 1);

            % sample from each block
            [samples, ix] = randp(prob, temp, size(temp,1), 1);
            ix_lin = sub2ind(size(temp),1:size(temp,1),ix');

            out = reshape( samples, size(in,1)/ratio, size(in,2)/ratio );
            indices = reshape( linind(ix_lin), size(in,1)/ratio, size(in,2)/ratio );

        elseif strcmp(operation, 'auto') == 1

            temp = nan(ratio*ratio, size(in,1)/ratio * size(in,2)/ratio);
            for k=1:ratio
                for m=1:ratio
                    temp2 = in(m:ratio:size(in,1), k:ratio:size(in,2));
                    temp((k-1)*ratio+m, :) = temp2(:)';
                end
            end

            % each column of 'temp' represents one 4x4 block =>
            % multiply each column by the shared weights and
            % add bias to every column, then reshape to matrix
            out = reshape( sum(temp .* repmat(weights(:),1,size(temp,2)),1) + bias, ...
                            size(in,1)/ratio, size(in,2)/ratio);
        end
    end
           
end