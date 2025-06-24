function [ selectedElectrodes ] = axonTrack( fileManagerObj, varargin )

p = inputParser;

p.addParameter('NumberOfElectrodes', 40);
p.addParameter('ActivityThreshold', 0.5);
p.addParameter('MinNeighbourElectrodes', 5);
p.addParameter('PlotDistributions', false);

p.parse(varargin{:});
args = p.Results;

rawXpos = fileManagerObj.processedMap.xpos./17.5;
rawYpos = fileManagerObj.processedMap.ypos./17.5;
spikeCount = mxw.activityMap.computeSpikeCount(fileManagerObj);
meanAmplitude = mxw.activityMap.computeMeanAmplitude(fileManagerObj);
amplitudeVarCoeff = mxw.util.computeAmplitudeVarCoeff(fileManagerObj, meanAmplitude);

ActivityThrValue = mxw.util.percentile(spikeCount, 100 - args.ActivityThreshold);
interestingElec = find(spikeCount > ActivityThrValue);

[~, idxAmpValues] = sort(meanAmplitude(interestingElec));
orderedInterestingElec = interestingElec(idxAmpValues);

if length(idxAmpValues) > args.NumberOfElectrodes
    temporalSelectedElectrodes = orderedInterestingElec(1:args.NumberOfElectrodes);
else
    temporalSelectedElectrodes = orderedInterestingElec;
end

populatedSelection = mxw.util.electrodeSelection.populateSelec(temporalSelectedElectrodes, rawXpos, rawYpos, args.MinNeighbourElectrodes);
electrodes = find(populatedSelection);
selectedElectrodes.electrodes = electrodes;
selectedElectrodes.xpos = fileManagerObj.processedMap.xpos(electrodes);
selectedElectrodes.ypos = fileManagerObj.processedMap.ypos(electrodes);

if args.PlotDistributions
    mxw.plot.selectionDistributions(spikeCount, meanAmplitude, amplitudeVarCoeff, temporalSelectedElectrodes);
end

disp([num2str(sum(populatedSelection)) ' electrodes for axon tracking; '...
    num2str(length(temporalSelectedElectrodes)) ' from ranked list']);
end

