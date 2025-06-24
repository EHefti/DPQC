function [ groups ] = electrodeGroups( fileManagerObjORmap, radius )
    % ELECTRODEGROUPS groups the electrodes given by fileManagerObjORmap. 
    % The groupping criteria is defined by the 'radius'.
    % 
    % [groups] = mxw.util.electrodeGroups(fileManagerObjORmap);
    %
    %   -The input parameters for this function are:
    %    -fileManagerObjORmap: object of the class 'mxw.fileManager' or
    %                          structure containing the 'xpos', 'ypos' 
    %                          and 'electrode' fields of a map
    %    -radius: radius (in um) used to group the electrodes
    %
    %   -The output parameter for this method is:
    %    -groups: cell array where each cell contains one electrode group
    %                                   
    %  -Examples
    %     -Considering we want to group the electrodes in the 
    %     'fileManRecording' object, making groups that contain electrodes 
    %     that are not more than 200um away from each other:
    %
    %     [groups] = mxw.util.electrodeGroups(fileManRecording, 200);
    %
    %
    
groups = {};
takenElectrodes = [];

if isa(fileManagerObjORmap, 'mxw.fileManager')
    xpos = fileManagerObjORmap.processedMap.xpos;
    ypos = fileManagerObjORmap.processedMap.ypos;
    electrode = fileManagerObjORmap.processedMap.electrode;
else
    xpos = fileManagerObjORmap.xpos;
    ypos = fileManagerObjORmap.ypos;
    electrode = fileManagerObjORmap.electrodes;
end

figure;
hold on;

while ~isempty(electrode)

    [~, index] = sort(sqrt(xpos.^2 + ypos.^2));

    topPointX = xpos(index(1));
    topPointY = ypos(index(1));

    diffCloseElec = sqrt((topPointX - xpos).^2 + (topPointY - ypos).^2);
    
    closeElectrodes = diffCloseElec < radius;

    scatter(xpos(closeElectrodes), ypos(closeElectrodes), 'filled')

    groups{end+1} = double(electrode(closeElectrodes));
    takenElectrodes = [takenElectrodes ; electrode(closeElectrodes)];
    
    xpos(closeElectrodes) = [];
    ypos(closeElectrodes) = [];
    electrode(closeElectrodes) = [];
end
end

