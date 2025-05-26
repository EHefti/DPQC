function [ index, amplitude ] = findPeaks(vector, varargin)
    % FINDPEAKS finds local maxima (peaks) in 'vector' that are below/above  
    % a certain negative/positive threshold
    % 
    % [index, amplitude] = mxw.util.findPeaks(vector);
    %
    %   -The input parameters for this function are:
    %    -vector: vector with values to process
    %    -varargin: ...
    %    -'NegativeThreshold': negative threshold
    %    -'PositiveThreshold': positive threshold
    %
    %   -The output parameters for this function are:
    %    -index: indices where the peaks were found (with respect to 
    %            'vector')
    %    -amplitude: amplitude of the peaks found
    %
    %  -Examples
    %     -Considering we want to find the peaks that are below -150uV 
    %     within the vector 'trace':
    %
    %     [index, amplitude] = mxw.util.findPeaks(trace, ...
    %       'NegativeThreshold', -150);
    %
    %     -Considering we want to find the burst peaks above 1000Hz in 
    %     'firingRate':
    %
    %     [index, amplitude] = mxw.util.findPeaks(firingRate, ...
    %       'PositiveThreshold', 1000);
    %
    %

p = inputParser;

p.addRequired('vector');
p.addParameter('NegativeThreshold', []);
p.addParameter('PositiveThreshold', []);

p.parse(vector, varargin{:});
args = p.Results;

differences = diff(vector);

if ~isempty(args.NegativeThreshold) 
    threshold = args.NegativeThreshold;
    peaks = ((differences(1:end-1,:) < 0) & (differences(2:end,:) > 0) & (vector(2:end-1,:) < threshold));

elseif ~isempty(args.PositiveThreshold)
    threshold = args.PositiveThreshold;
    peaks = ((differences(1:end-1,:) > 0) & (differences(2:end,:) < 0) & (vector(2:end-1,:) > threshold));
    
else
    error('Please set a threshold')
end

dummyFill = zeros(size(threshold));
peaks = [dummyFill ; peaks ; dummyFill];

[index, ~, amplitude] = find(vector.*peaks);
end