function [ values ] = computeMeanAmplitude( fileManagerObj )
    % COMPUTEMEANAMPLITUDE computes the mean of the spike amplitudes in 
    % each electrode along all the recordings in 'fileManagerObj'. The
    % spike amplitudes are the ones previously saved by the recording
    % software every time a spike was detected.
    % 
    % meanAmp = mxw.activityMap.computeMeanAmplitude(fileManagerObj);
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
        tempValues(i) = mean(fileManagerObj.extractedSpikes(iFile).amplitude{i});
    end
    
    tempValues(isnan(tempValues)) = 0;
    values = values + tempValues;
end
end
