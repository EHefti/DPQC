function [ selectedElectrodes ] = networkRec( fileManagerObj, varargin )
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

p.addParameter('NumberOfElectrodes', 1024);
p.addParameter('ActivityThreshold', 10);
p.addParameter('MinNeighbourElectrodes', 2);
p.addParameter('AmplitudeThreshold', -25);
p.addParameter('PlotDistributions', false);

p.parse(varargin{:});
args = p.Results;

rawXpos = fileManagerObj.processedMap.xpos./17.5;
rawYpos = fileManagerObj.processedMap.ypos./17.5;
spikeCount = mxw.activityMap.computeSpikeCount(fileManagerObj);
meanAmplitude = mxw.activityMap.computeMeanAmplitude(fileManagerObj);
amplitudeVarCoeff = mxw.util.computeAmplitudeVarCoeff(fileManagerObj, meanAmplitude);

ActivityThrValue = mxw.util.percentile(spikeCount, 100 - args.ActivityThreshold);
interestingElec = find(spikeCount > ActivityThrValue & meanAmplitude < args.AmplitudeThreshold);

[~, idxStdValues] = sort(amplitudeVarCoeff(interestingElec));
orderedInterestingElec = interestingElec(idxStdValues);

totalElec = length(fileManagerObj.processedMap.electrode);
minNumElec = min(args.NumberOfElectrodes, length(orderedInterestingElec));
electrodes = [];

while (totalElec > args.NumberOfElectrodes)
    temporalSelectedElectrodes = orderedInterestingElec(1:minNumElec);
    
    populatedSelection = mxw.util.electrodeSelection.populateSelec(temporalSelectedElectrodes, ...
        rawXpos, rawYpos, args.MinNeighbourElectrodes);
    
    electrodes = find(populatedSelection);
    totalElec = length(electrodes);
    
    excessElec = totalElec - args.NumberOfElectrodes;
    
    if excessElec >= 4
        minNumElec = minNumElec - round(excessElec/4);
    else
        minNumElec = minNumElec-1;
    end
end

selectedElectrodes.electrodes = electrodes;
selectedElectrodes.xpos = fileManagerObj.processedMap.xpos(electrodes);
selectedElectrodes.ypos = fileManagerObj.processedMap.ypos(electrodes);

if args.PlotDistributions
    mxw.plot.selectionDistributions(spikeCount, meanAmplitude, amplitudeVarCoeff, temporalSelectedElectrodes);
end

disp([num2str(totalElec) ' electrodes in selection; ' num2str(minNumElec) ' from ranked list']);
end
