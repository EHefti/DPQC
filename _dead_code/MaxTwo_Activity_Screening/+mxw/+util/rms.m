function [ value ] = rms( vector )
    % RMS calculates the root mean squared value in 'vector'.
    % 
    % [value] = mxw.util.rms(vector);
    %
    %   -The input parameter for this function is:
    %    -vector: vector with values of interest 
    %
    %   -The output parameter for this function is:
    %    -value: value with the computed rms
    %
    %  -Examples
    %     -Considering we want to compute the rms value of the vector 
    %     'amplitude':
    %
    %     rmsValue = mxw.util.rms(amplitude);
    %
    %

  value = sqrt(mean(vector.* conj(vector)));
end

