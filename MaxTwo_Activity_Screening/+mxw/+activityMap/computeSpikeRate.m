function [ values ] = computeSpikeRate( fileManagerObj )
    % COMPUTESPIKERATE computes the spiking rate in every electrode along
    % all the recordings in 'fileManagerObj'. The spiking rate is computed 
    % by counting the number of spikes per electrode per recording, and 
    % then dividing the count by the recording length.
    % 
    % spikeRate = mxw.activityMap.computeSpikeRate(fileManagerObj);
    % 
    %   -The input parameter for this function is:
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    
    %   -The output parameter for this function is:
    %    -values: vector containing the spiking rate of each electrode 
    %    
    %

if fileManagerObj.version == '20160704'    
    nFiles = fileManagerObj.nFiles;
elseif fileManagerObj.version == '20190530'
    nFiles = fileManagerObj.nRecordings;
end
    
values = zeros(length(fileManagerObj.processedMap.electrode), 1);

for iFile = 1:nFiles
    values = values + (cellfun('length', fileManagerObj.extractedSpikes(iFile).amplitude) * fileManagerObj.fileObj(iFile).samplingFreq...
        /fileManagerObj.fileObj(iFile).dataLenSamples);
end
end

