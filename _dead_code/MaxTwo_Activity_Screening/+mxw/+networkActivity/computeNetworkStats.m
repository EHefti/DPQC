function [ values ] = computeNetworkStats( networkActivityVector, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

p.addRequired('struct', @(x) isstruct(x));
p.addParameter('Threshold', 1.35);

p.parse(networkActivityVector, varargin{:});
args = p.Results;

rmsFiringRate = mxw.util.rms(networkActivityVector.firingRate);

% check toolbox
% result = license('test','Signal Processing Toolbox');
% %             if exist('butter')
% if result
%     % if exist('findPeaks')
%     [maxAmplitudesValues, maxAmplitudesTimes] = findPeaks(networkActivityVector.firingRate,...
%         networkActivityVector.time, 'MinPeakHeight', args.Threshold * rmsFiringRate);
% else
    [tmp_times, maxAmplitudesValues] = mxw.util.findPeaks(networkActivityVector.firingRate,...
        'PositiveThreshold', args.Threshold * rmsFiringRate);
    %
    maxAmplitudesTimes = networkActivityVector.time(tmp_times);
% end
values.maxAmplitudesValues = maxAmplitudesValues;

maxAmplitudeTimeDiff = diff(maxAmplitudesTimes);
values.maxAmplitudeTimeDiff = maxAmplitudeTimeDiff;
values.maxAmplitudesTimes = maxAmplitudesTimes;
values.maxAmplitudesValues = maxAmplitudesValues;


end