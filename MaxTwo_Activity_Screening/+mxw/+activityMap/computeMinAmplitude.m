function [ values ] = computeMinAmplitude( fileManagerObj )
    % COMPUTEMEANAMPLITUDE computes the minimum amplitude in 
    % each electrode along all the recordings in 'fileManagerObj'. The
    % spike amplitudes are the ones previously saved by the recording
    % software every time a spike was detected.
    % 
    % minAmp = mxw.activityMap.computeMinAmplitude(fileManagerObj);
    % 
    %   -The input parameter for this function is:
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    
    %   -The output parameter for this function is:
    %    -values: vector containing the mean of the spike amplitudes in
    %             each electrode
    %    
    %

nFiles = fileManagerObj.nFiles;
nElectrodes = length(fileManagerObj.processedMap.electrode);
tempValues = zeros(nElectrodes,1);
values = zeros(nElectrodes,1);

for iFile = 1:nFiles
    for i = 1:nElectrodes
        minVal = min(fileManagerObj.extractedSpikes(iFile).amplitude{i});
        if isempty(minVal)
            minVal = 0;
        end
        tempValues(i) = minVal;
    end
    
    values = values + tempValues;
end
end
