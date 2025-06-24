function [ value ] = percentile( vector, prcn )
    % PERCENTILE calculates the 'percentile' value from the distribution of
    % values in 'vector'.
    % 
    % [value] = mxw.util.percentile(vector, percentile);
    %
    %   -The input parameters for this function are:
    %    -vector: vector with values of interest
    %    -prcn: percentile to calculate from 'vector' 
    %
    %   -The output parameter for this function is:
    %    -value: value with the percentile computed
    %
    %  -Examples
    %     -Considering we want to know the 80 percentile value of the
    %     'spikeCount' vector:
    %
    %     spikeCountPcrn80 = mxw.util.percentile(spikeCount, 80);
    %
    %
    
    if isempty(vector)
        value = NaN;
    
    elseif length(vector) == 1
        value = vector;
        
    else
        linVector = linspace(0.5/length(vector), 1-0.5/length(vector), length(vector));
        value = interp1(linVector', sort(vector), prcn*0.01, 'spline');
    end
end