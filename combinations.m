function out = combinations(vec1, vec2, combnum)
%
% INPUTS
% vec1 : N x 1 vector
% vec2 : N x 1 vector
% combnum : number to select (row length of out)
%
% OUTPUTS
% out : N*N x combnum matrix
%

out = nan(length(vec1)*length(vec2),2);

for m=1:length(vec2)
    for k=1:length(vec1)
        out((m-1)*combnum+k,:) = [vec1(k), vec2(m)];
    end
end


end