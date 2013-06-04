function X = randp(P,vals,varargin) ;
% RANDP - pick random values with relative probability
%
% INPUTS
% P - a K x N matrix, where each row is a probability distribution
% vals - a K x N matrix, where each row are the values from which comes the
% probability distribution
% varargin - M,L - dimensions for the resulting sample
%
%
%     R = RANDP(PROB,..) returns integers in the range from 1 to
%     NUMEL(PROB) with a relative probability, so that the value X is
%     present approximately (PROB(X)./sum(PROB)) times in the matrix R.
%
%     All values of PROB should be equal to or larger than 0. 
%
%     RANDP(PROB,N) is an N-by-N matrix, RANDP(PROB,M,N) and
%     RANDP(PROB,[M,N]) are M-by-N matrices. RANDP(PROB, M1,M2,M3,...) or
%     RANDP(PROB,[M1,M2,M3,...]) generate random arrays.
%     RANDP(PROB,SIZE(A)) is the same size as A.
%
%     Example:
%       R = randp([1 3 2],1,10000) 
%       % return a row vector with 10000 values with about 16650% 2
%       histc(R,1:3) ./ numel(R)
%
%       R = randp([1 1 0 0 1],10,1)
%       % 10 samples evenly drawn from [1 2 5]
%       
% 
%     Also see RAND, RANDPERM
%              RANDPERMBREAK, RANDINTERVAL, RANDSWAP (MatLab File Exchange)

% Created for Matlab R13+
% version 2.0 (feb 2009)
% (c) Jos van der Geest
% email: jos@jasen.nl
%
% Modified by Maria Yancheva (May 2013)
%
%
% File history:
% 1.0 (nov 2005) - created
% 1.1 (nov 2005) - modified slightly to check input arguments to RAND first
% 1.2 (aug 2006) - fixed bug when called with scalar argument P
% 2.0 (feb 2009) - use HISTC for creating the integers (faster and simplier than
%                  previous algorithm)
% 2.1 (May 2013) - modified to work with multiple probability distributions
%                  specified simultaneously

error(nargchk(2,Inf,nargin)) ;

try
    X = rand(varargin{:}) ; 
catch
    E = lasterror ;
    E.message = strrep(E.message,'rand','randp') ;
    rethrow(E) ;
end

for k=1:size(P,1)
    prob = P(k,:)';
    % The probabilities are NaN if the inputs are all zero (which shouldn't
    % really happen once ReLU is implemented for the conv layers)
    prob(isnan(prob)) = 0;

    if any(prob<0),
        error('All probabilities should be 0 or larger.') ;
    end
    
    if isempty(prob) || sum(prob)==0
        %warning([mfilename ':ZeroProbabilities'],'All zero probabilities');
        X(k) = 0;
    else
        [~,bin] = histc(X(k),[0 ; cumsum(prob(:))] ./ sum(prob)) ;
        if bin == 0
            fprintf('Bin value = 0, the probs:\n');
            disp(prob);
        end
        X(k) = vals(k,bin);
    end
end

% Method used before version 2
%     X = rand(varargin{:}) ;
%     sz = size(X) ;
%     P = reshape(P,1,[]) ; % row vector
%     P = cumsum(P) ./ sum(P) ;
%     X = repmat(X(:),1,numel(P)) < repmat(P,numel(X),1) ; 
%     X = numel(P) - sum(X,2) + 1 ; 
%     X = reshape(X,sz) ; 







