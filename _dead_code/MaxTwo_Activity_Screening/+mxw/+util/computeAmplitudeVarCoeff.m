function [ values ] = computeAmplitudeVarCoeff( fileManagerObj, amplitudeValue )
    % COMPUTEAMPLITUDEVARCOEFF computes how much do the spike amplitudes in 
    % each electrode vary along the whole recording in the 'fileManagerObj'
    % object. This function is internally needed in the 
    % 'mxw.util.electrodeSelection...' functions. Therefore, the user 
    % should not directly call or change this function.
    %
    %

nFiles = fileManagerObj.nFiles;
nElectrodes = length(fileManagerObj.processedMap.electrode);
tempValues = zeros(nElectrodes,1);
values = zeros(nElectrodes,1);

for iFile = 1:nFiles
    for i = 1:nElectrodes
        tempValues(i) = var(fileManagerObj.extractedSpikes(iFile).amplitude{i})/abs(amplitudeValue(i));
    end
    
    tempValues(isnan(tempValues)) = 0;
    values = values + tempValues;
end
end