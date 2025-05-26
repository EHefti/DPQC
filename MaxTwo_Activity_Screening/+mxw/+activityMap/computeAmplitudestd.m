function [ values ] = computeAmplitudestd( fileManagerObj )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

nFiles = fileManagerObj.nFiles;
nElectrodes = length(fileManagerObj.processedMap.electrode);
tempValues = zeros(nElectrodes,1);
values = zeros(nElectrodes,1);

for iFile = 1:nFiles
    for i = 1:nElectrodes
        tempValues(i) = std(fileManagerObj.extractedSpikes(iFile).amplitude{i});
    end
    
    tempValues(isnan(tempValues)) = 0;
    values = values + tempValues;
end
end