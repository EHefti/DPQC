function [ newElectrodesArray, newFilesArray ] = electrodes2Files( electrodesArray, filesArray )
    % ELECTRODES2FILES receives an array of electrodes and a cell array 
    % with the files coresponding to those electrodes, and returns and
    % array of files, and a cell array of the electrodes corresponding to
    % those files. This is an internal helper function to extract the data
    % from the 'mxw.fileManager' object and should not be called or 
    % modified by the user
    % 
    %

nFiles = length(unique(cell2mat(filesArray)));
nElectrodes = size(electrodesArray, 2);
newFilesArray = zeros(1, nFiles);
newElectrodesArray = cell(1, nFiles);
j = 0;
index = 1;

for i = 1:nElectrodes
    index = index+j;
    for j = 1:length(filesArray{i})
        repeatedFile = find(newFilesArray == filesArray{i}(j));
        
        if isempty(repeatedFile)
            newElectrodesArray{index+j-1} = electrodesArray(i);
            newFilesArray(index+j-1) = filesArray{i}(j) ;
        else
            
            newElectrodesArray{repeatedFile} = [newElectrodesArray{repeatedFile}, electrodesArray(i)];
            index = index - 1;
        end
    end
end
end

