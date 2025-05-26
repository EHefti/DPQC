function [ values ] = computeAmplitude90percentile( fileManagerObj )
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
if fileManagerObj.version == '20160704'    
    nFiles = fileManagerObj.nFiles;
elseif fileManagerObj.version == '20190530'
    nFiles = fileManagerObj.nRecordings;
end
    
nElectrodes = length(fileManagerObj.processedMap.electrode);
tempValues = zeros(nElectrodes,1);
values = zeros(nElectrodes,1);

for iFile = 1:nFiles
    for i = 1:nElectrodes
        tempValues(i) = mxw.util.percentile(fileManagerObj.extractedSpikes(iFile).amplitude{i}, 10);
    end
    
    tempValues(isnan(tempValues)) = 0;
    values = values + tempValues;
end
end

