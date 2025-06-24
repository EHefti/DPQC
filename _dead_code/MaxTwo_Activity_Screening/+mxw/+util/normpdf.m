function [ normalDist ] = normpdf( vector, mean, sigma )
    % NORMPDF computes the probability density function (pdf) of the normal 
    % distribution with 'mean' and 'sigma' as mean and standard deviation
    % respectively, evaluated at the value points in 'vector'
    % 
    % [normalDist] = mxw.util.normpdf(vector, mean, sigma);
    %
    %   -The input parameters for this function are:
    %    -vector: vector with values to compute the distribution
    %    -mean: mean of the distribution
    %    -sigma: standard deviation of the distribution
    %
    %   -The output parameter for this function is:
    %    -normalDist: computed probability density function
    %
    %  -Examples
    %     -Considering we want a pdf with mean equal to zero and standard
    %     deviation of 3 computed over the values of 'Xvector':
    %
    %     normalDist = mxw.util.normpdf(Xvector, 0, 3);
    %
    %

normalDist = exp(-0.5 * ((vector - mean)./sigma).^2) ./ (sqrt(2*pi) .* sigma);
end
