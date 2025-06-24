function [ values ] = computeISIv( fileManagerObj )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

nFiles = fileManagerObj.nFiles;
nElectrodes = length(fileManagerObj.processedMap.electrode);
tempValues = zeros(nElectrodes,1);
values = zeros(nElectrodes,1);

for iFile = 1:nFiles
    for i = 1:nElectrodes
        tempValues(i) = (sum(diff(fileManagerObj.extractedSpikes(iFile).frameno{i})<40)/length(fileManagerObj.extractedSpikes(iFile).frameno{i}))*100;

    end
    
    tempValues(isnan(tempValues)) = 0;
    values = values + tempValues;
end
end