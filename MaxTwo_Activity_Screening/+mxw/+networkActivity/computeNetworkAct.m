function [ values ] = computeNetworkAct( fileManagerORspikeTimes, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

p.addRequired('struct', @(x) isobject(x) || isstruct(x));
p.addParameter('file', []);
p.addParameter('BinSize', 0.01);
p.addParameter('GaussianSigma', 0.3);  % kernel standard deviation in s
p.addParameter('MinValue', 0);
p.addParameter('MaxValue', []);

p.parse(fileManagerORspikeTimes, varargin{:});
args = p.Results;

if isa(fileManagerORspikeTimes, 'mxw.fileManager')
    relativeSpikeTimes = mxw.util.computeRelativeSpikeTimes(fileManagerORspikeTimes, 'file', args.file);
    if size(fileManagerORspikeTimes.rawMap,2)>1
        total_channels = length(unique(fileManagerORspikeTimes.rawMap(args.file).spikes.channel));
    else
        total_channels = length(unique(fileManagerORspikeTimes.rawMap.spikes.channel));
    end
else
    relativeSpikeTimes = fileManagerORspikeTimes;
    total_channels = length(unique(fileManagerORspikeTimes.channel));
end

if isempty(args.MaxValue)
    args.MaxValue = max(relativeSpikeTimes.time);
end

kernel = mxw.util.normpdf(-3*args.GaussianSigma : args.BinSize : 3*args.GaussianSigma, 0, args.GaussianSigma); % gaussian kernel
kernel = kernel * args.BinSize;

timeVector  = args.MinValue:args.BinSize:args.MaxValue;
binnedTimes = histc(relativeSpikeTimes.time, timeVector);

% gaussian kernel and convolution
firingRate = conv(binnedTimes, kernel, 'same');
firingRate = firingRate / args.BinSize;

values.time = timeVector;
values.firingRate = firingRate/total_channels;
end