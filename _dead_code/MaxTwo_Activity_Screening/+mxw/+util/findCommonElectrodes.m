function [ commonElectrodes ] = findCommonElectrodes( fileManagerObj )
    % FINDCOMMONELECTRODES finds the electrodes that are common in all the 
    % files in 'fileManagerObj', i.e. the electrodes that are always routed
    % along all the recording files. This is a handful function to find the
    % fixed electrodes used for axonal tracking
    % 
    % [commonElectrodes] = mxw.util.findCommonElectrodes(fileManagerObj);
    %
    %   -The input parameter for this function is:
    %    -fileManagerObj: object of the class 'mxw.fileManager' 
    %
    %

electrodes = fileManagerObj.processedMap.electrode(...
    cellfun('length', fileManagerObj.processedMap.fileIndex) == fileManagerObj.nFiles);
index = mod(find(fileManagerObj.processedMap.electrode == electrodes'), ...
    length(fileManagerObj.processedMap.electrode));
index(index == 0) = length(fileManagerObj.processedMap.electrode);

xpos = fileManagerObj.processedMap.xpos(index);
ypos = fileManagerObj.processedMap.ypos(index);

commonElectrodes.electrodes = electrodes;
commonElectrodes.xpos = xpos;
commonElectrodes.ypos = ypos;
end

