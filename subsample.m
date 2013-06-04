 function out = subsample(in, ratio, operation, weights, bias) 
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
 % 
 
         % By default, perform average
         if nargin < 3
            operation = 'average';
         end
 
        %Сжатие изображений простым усреднением
        %in - входная матрица, ratio - во сколько раз сжать, out - выходная
        %матрица, ws - матрица весов для субдискретизации, должна быть
        %размером равным размеру карты после субдискретизации, bs - матрица
        %смещений, по размерности такая же как ws
        switch ratio
            case 4
                if strcmp(operation, 'average') == 1
                    out = 0;
                    for k=1:4
                        for l=1:4
                            out = out+in(1+(k-1):4:size(in,1),1+(l-1):4:size(in,2));
                        end
                    end

                    out = out/16;
                elseif strcmp(operation, 'max') == 1
                    % each column of 'temp' contains one ratio x ratio block
                    % (in row-major order, i.e. column 2 represents block
                    % on row 2, col 1 of the original input matrix)
                    temp = nan(ratio*ratio, size(in,1)/ratio * size(in,2)/ratio);
                    for k=1:4
                        for m=1:4
                            temp2 = in(m:ratio:size(in,1), k:ratio:size(in,2));
                            temp((k-1)*ratio+m, :) = temp2(:)';
                        end
                    end
                    
                    out = reshape( max(temp, [], 1), size(in,1)/ratio, size(in,2)/ratio );
                    
                elseif strcmp(operation, 'stochastic') == 1
                    temp = nan(ratio*ratio, size(in,1)/ratio * size(in,2)/ratio); % 16 x 1
                    for k=1:4
                        for m=1:4
                            temp2 = in(m:ratio:size(in,1), k:ratio:size(in,2));
                            temp((k-1)*ratio+m, :) = temp2(:)';
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
                    out = reshape( randp(prob', temp', size(temp,2), 1), size(in,1)/ratio, size(in,2)/ratio );
                elseif strcmp(operation, 'auto') == 1
                    
                    temp = nan(ratio*ratio, size(in,1)/ratio * size(in,2)/ratio);
                    for k=1:4
                        for m=1:4
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
            case 2
                if strcmp(operation, 'average') == 1
                    out = (in(1:ratio:size(in,1),1:ratio:size(in,2)) + in(1:ratio:size(in,1),2:ratio:size(in,2))...
                        +in(2:ratio:size(in,1),1:ratio:size(in,2)) + in(2:ratio:size(in,1),2:ratio:size(in,2)))./(ratio*ratio);
                elseif strcmp(operation, 'max') == 1
                    a = in(1:ratio:size(in,1),1:ratio:size(in,2));
                    b = in(2:ratio:size(in,1),1:ratio:size(in,2));
                    c = in(1:ratio:size(in,1),2:ratio:size(in,2));
                    d = in(2:ratio:size(in,1),2:ratio:size(in,2));
                    temp = [a(:), b(:), c(:), d(:)];
                    out = reshape( max(temp, [], 2), size(in,1)/ratio, size(in,2)/ratio );
                elseif strcmp(operation, 'stochastic') == 1
                    
                    % Build a multinomial distribution for each of the
                    % regions to be sampled
                    a = in(1:ratio:size(in,1),1:ratio:size(in,2));
                    b = in(2:ratio:size(in,1),1:ratio:size(in,2));
                    c = in(1:ratio:size(in,1),2:ratio:size(in,2));
                    d = in(2:ratio:size(in,1),2:ratio:size(in,2));
                    temp = [a(:), b(:), c(:), d(:)];
                    
                    % Assume all inputs are positive or zero! This is
                    % handled by using ReLU activation on the conv layers,
                    % but just in case handle it here:
                    temp(temp < 0) = 0;
                    
                    % each row of 'temp' represents one 2x2 block =>
                    % convert the values in each block into probabilities
                    prob = temp ./ repmat(sum(temp,2), 1, size(temp,2));
                    
                    % sample from each block
                    out = reshape( randp(prob, temp, size(temp,1), 1), size(in,1)/ratio, size(in,2)/ratio );
                elseif strcmp(operation, 'auto') == 1
                    
                    % Multiple each (ratio x ratio) region by its learned
                    % weights and add bias -> this corresponds to learning
                    % the pooling function instead of hardcoding it
                    a = in(1:ratio:size(in,1),1:ratio:size(in,2));
                    b = in(2:ratio:size(in,1),1:ratio:size(in,2));
                    c = in(1:ratio:size(in,1),2:ratio:size(in,2));
                    d = in(2:ratio:size(in,1),2:ratio:size(in,2));
                    temp = [a(:), b(:), c(:), d(:)]; % 16 x 4
                    
                    % each row of temp represents one 2x2 block
                    out = reshape( sum(temp .* repmat(weights(:)',size(temp,1),1),2) + bias, ...
                                    size(in,1)/ratio, size(in,2)/ratio );
                end
            case 1
                out = in; %Пустышка - нужно, если начинаем со сверточного слоя
            otherwise
                disp('Unsupported ratio');
        end
%Убрал, т.к. при обратном распространении нам требуется карта признаков еще
%не умноженная на веса        
%    out = out.*ws + bs;        
end