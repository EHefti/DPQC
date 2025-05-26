function [ values ] = computeSpikeCount( fileManagerObj )
    % COMPUTESPIKECOUNT counts the number of spikes in every electrode
    % along all the recordings in 'fileManagerObj'. The spikes counted
    % are the ones previously detected by the recording software.
    % 
    % spikeCount = mxw.activityMap.computeSpikeCount(fileManagerObj);
    % 
    %   -The input parameter for this function is:
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    
    %   -The output parameter for this function is:
    %    -values: vector containing the total number of spikes in each 
    %             electrode
    %    
    %
    
nFiles = fileManagerObj.nFiles;
values = zeros(length(fileManagerObj.processedMap.electrode), 1);

for iFile = 1:nFiles
    values = values + cellfun('length', fileManagerObj.extractedSpikes(iFile).amplitude);
end
end

