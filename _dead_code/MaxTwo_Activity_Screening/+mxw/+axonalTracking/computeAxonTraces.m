function [ axonTraces, electrodeGroups, ts, w, s ] = computeAxonTraces( fileManagerObj, axonTrackElec, varargin )
    % COMPUTEAXONTRACES computes the axon traces from the neurons on top of
    % the electrodes in 'axonTrackElec'. The first step to compute the 
    % traces is to cluster the electrodes in 'axonTrackElec' to provide the 
    % 'electrodeGroups'. Then the function goes through every file  
    % extracting chunks of data. For each extracted data chunk, the
    % function takes each group of electrodes in 'electrodesGroups' 
    % looking for spikes that cross the threshold defined by 
    % 'SpikeDetThreshold'. Whenever a spike is detected in the group of 
    % electrodes under consideration, the function extracts cutouts of data
    % in the rest of the electrodes in that file. The cutouts corresponding
    % to each group of electrodes are saved independently. For each file, 
    % the cutouts, among all the electrodes, corresponding to the same
    % electrode group are averaged, the averaged value is called waveform.
    % Therefore, after detecting the spikes in each group of electrodes
    % among all data chunks in every file and averaging them, there is a 
    % waveform in every electrode corresponding to each group of
    % electrodes.
    % 
    % [axonTraces, electrodeGroups, spikesArray] = ...
    %   mxw.axonalTracking.computeAxonTraces(fileManagerObj, ...
    %     axonTrackElec)
    %
    %   -The input parameters for this function are:
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    -axonTrackElec: structure containing the number of the electrodes,
    %     and their 'x' and 'y' coordinates, which are defined as 
    %     interesting electrodes for being under highly active somas
    %    -varargin: ...
    %    -'SpikeDetThreshold': threshold in uV to be used for detection
    %    -'SecondsToLoadPerIteration': seconds, i.e. chunk size, to load
    %                                  per iteration from each file
    %    -'TotalSecondsToLoad': seconds to load, in total, from each file
    %    -'MaxDistClustering': maximum distance in, um, to consider when 
    %                          clustering the group of electrodes
    %    -'PrePointsSpike': number of sample points that are 
    %                       loaded from every electrode, to build the
    %                       cutouts, before the sample when the spike was
    %                       detected
    %    -'PostPointsSpike': number of sample points that are
    %                       loaded from every electrode, to build the
    %                       cutouts, after the sample when the spike was
    %                       detected
    %
    %   -The output parameters for this method are:
    %    -axonTraces: struct containing the 'map' and the 'traces'. The 
    %                 'map' is composed by the electrode numbers and 'x'
    %                 and 'y' coordinates of all the electrodes used among
    %                 all the files evaluated. The 'traces' is a cell array
    %                 cntaining one cell for each group in
    %                 'electrodeGroups', each cell contains the waveforms
    %                 over all the electrodes over all the files evaluated
    %    -electrodeGroups: array of cells containing the groups of
    %                      electrodes. Each number in each cell corresponds
    %                      to the index in the input 'axonTrackElec'
    %    -spikesArray: cell array where each column represents a file and 
    %                  each row a group from 'electrodeGroups'. Inside each
    %                  cell there is one row for each chunk of data loaded,
    %                  where the structure contains information about the
    %                  detected spikes, and the number on its right is the
    %                  first data sample of that chunk
    %                                   
    %  -Examples
    %     -Considering we want to compute the axon traces of a folder that
    %     contains several recordings. We want to find the waveforms these
    %     axons are generating over tho whole MEA in a period of 60
    %     seconds of data:
    %
    %     [axonTraces, electrodeGroups, spikesArray] = ...
    %       mxw.axonalTracking.computeAxonTraces(fileManagerObj, ...
    %       axonTrackingElec, 'SecondsToLoadPerIteration', 30, ...
    %       'TotalSecondsToLoad', 60);
    %
    %     -Now we want to compute the axon traces but along the whole 
    %     recording, considering a threshold of -35uV for detection and 
    %     cutting out 30 samples before and 70 samples after each spike
    %     respectively:
    %
    %     [axonTraces, electrodeGroups, spikesArray] = ...
    %       mxw.axonalTracking.computeAxonTraces(fileManagerObj, ...
    %       axonTrackingElec, 'SpikeDetThreshold', -35, 'PrePointsSpike',
    %       30, 'PostPointsSpike', 70);
    %
    %
    
p = inputParser;

p.addParameter('SpikeDetThreshold', 6);
p.addParameter('SecondsToLoadPerIteration', 20);
p.addParameter('TotalSecondsToLoad', 'full');
p.addParameter('MaxDistClustering', 20);
p.addParameter('PrePointsSpike', 20);
p.addParameter('PostPointsSpike', 30);

p.parse(varargin{:});
args = p.Results;

if ~(strcmp(args.TotalSecondsToLoad, 'full'))
    if args.SecondsToLoadPerIteration > args.TotalSecondsToLoad
        error('SecondsToLoadPerIteration has to be equal or less than TotalSecondsToLoad')
    end
end

nFiles = fileManagerObj.nFiles;
sampFreq = fileManagerObj.fileObj(1).samplingFreq;
secondsToLoadPerIteration = args.SecondsToLoadPerIteration;
temporalAxonTraces = cell(nFiles,1);

[~, ~, ~, electrodeGroups] = clusterXYpoints([axonTrackElec.xpos axonTrackElec.ypos], args.MaxDistClustering, 1);

for iFile = 1:nFiles
    disp(iFile)
    
    if strcmp(args.TotalSecondsToLoad, 'full')
        totalLengthSamples = fileManagerObj.fileObj(iFile).dataLenSamples;
    else
        totalLengthSamples = args.TotalSecondsToLoad * sampFreq;
    end
    
    chunkSize = 1:sampFreq*secondsToLoadPerIteration:totalLengthSamples;
    fullWaveforms = cell(size(electrodeGroups));
    waveformCount = zeros(size(electrodeGroups));
    
    timestamps{iFile} = cell(size(electrodeGroups));
    waveforms_ums{iFile} = cell(size(electrodeGroups));
    std_ums{iFile} = cell(size(electrodeGroups));
    
    for iChunk = 1:length(chunkSize)
        startPoint = chunkSize(iChunk);
        endPoint = min(sampFreq*secondsToLoadPerIteration, totalLengthSamples - startPoint + 1);
        
        [completeData, filesArray, electrodesArray] = fileManagerObj.extractBPFData(startPoint, endPoint, 'files', iFile);
        normCompleteData = bsxfun(@minus, completeData, round(mean(completeData)));
        
        for iElecGroup = 1:size(electrodeGroups,1)
            currentElectrodes = axonTrackElec.electrodes(electrodeGroups{iElecGroup})';
            
            index = mod(find(cell2mat(electrodesArray)' == currentElectrodes), length(cell2mat(electrodesArray)));
            index(index == 0) = length(cell2mat(electrodesArray));
            dataCommonElec = completeData(:,index);
            
            spikes = ss_default_params(sampFreq);
            spikes.params.thresh = args.SpikeDetThreshold;
            disp(std(dataCommonElec)*spikes.params.thresh);
            spikes = ss_detect({dataCommonElec}, spikes);

            
            detectedSpikes = round(spikes.spiketimes*sampFreq);
            detectedSpikes = sort(detectedSpikes);
            detectedSpikes(detectedSpikes < 150) = [];
            detectedSpikes(detectedSpikes > length(dataCommonElec)-200) = [];
            
            timestamps{iFile}{iElecGroup} = [timestamps{iFile}{iElecGroup} spikes.spiketimes+startPoint/20000];
            waveforms_ums{iFile}{iElecGroup} = [waveforms_ums{iFile}{iElecGroup}; spikes.waveforms];
            std_ums{iFile}{iElecGroup} = [std_ums{iFile}{iElecGroup};  spikes.info.detect.thresh];
            
            prePointsSpike = args.PrePointsSpike;
            postPointsSpike = args.PostPointsSpike;
            
            if isempty(fullWaveforms{iElecGroup})
                fullWaveforms{iElecGroup}(:, :) = zeros(prePointsSpike + postPointsSpike +1, length(electrodesArray{1,1}));
            end
            
            for iSpike = 1:length(detectedSpikes)
                waveforms = single(normCompleteData(detectedSpikes(iSpike) - prePointsSpike : detectedSpikes(iSpike) + postPointsSpike, :));
%                 normWaveforms = bsxfun(@minus, waveforms, round(mean(waveforms)));
                
                fullWaveforms{iElecGroup}(:, :) = fullWaveforms{iElecGroup}(:, :) + waveforms;
                waveformCount(iElecGroup) = waveformCount(iElecGroup) + 1;
            end
        end
    end
    
    averagedFullWaveforms = fullWaveforms;
    waveformCount(waveformCount == 0) = 1;
    
    for i = 1:length(electrodeGroups)
        averagedFullWaveforms{i} = fullWaveforms{i}/waveformCount(i);
    end
    
    temporalAxonTraces{iFile} = averagedFullWaveforms;
end

axonTraces.map = [];
axonTraces.traces = [];

temporalTotalElectrodes = cell(nFiles,1);
temporalTotalX = cell(nFiles,1);
temporalTotalY = cell(nFiles,1);

for iFile = 1:nFiles
    temporalTotalElectrodes{iFile} = double(fileManagerObj.rawMap(iFile).map.electrode);
    temporalTotalX{iFile} = fileManagerObj.rawMap(iFile).map.x;
    temporalTotalY{iFile} = fileManagerObj.rawMap(iFile).map.y;
end

totalElectrodes = cell2mat(temporalTotalElectrodes);
totalX = cell2mat(temporalTotalX);
totalY = cell2mat(temporalTotalY);

[~, indicesFinalElec, ~] = unique(totalElectrodes);

electrode = totalElectrodes(indicesFinalElec);
x = totalX(indicesFinalElec);
y = totalY(indicesFinalElec);

axonTraces.map.electrode = electrode;
axonTraces.map.x = x;
axonTraces.map.y = y;

temporalTotalTraces = cell(length(temporalAxonTraces),1)';
traces = cell(length(temporalAxonTraces{1,1}),1);

for i = 1:length(temporalAxonTraces{1,1})
    for j = 1:length(temporalAxonTraces)
        temporalTotalTraces{1,j} = temporalAxonTraces{j,1}{i,1};
    end
    
    totalTraces = cell2mat(temporalTotalTraces);
    traces{i,1} = totalTraces(:,indicesFinalElec);
end

axonTraces.traces = traces;
ts = timestamps;
w = waveforms_ums;
s = std_ums;

end
