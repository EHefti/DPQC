function generateProbe( fileName, fileManagerObj, group, radius )
    % GENERATEPROBE creates a file containing the channels, and their 
    % coordinates, in 'group' representing a recording probe. There is no
    % output for this function except for the file created.
    % 
    % [values] = mxw.util.generateProbe(fileName, fileManagerObj, ...
    %               group, radius );
    %
    %   -The input parameters for this function are:
    %    -fileName: path and name of file containing the probe's channels
    %    and coordinates 
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    -group: one of the electrode groups created in 
    %            'mxw.util.electrodeGroups'
    %    -radius: radius to create the electrode group (defined in 
    %             'mxw.util.electrodeGroups')
    %                                                
    %  -Examples
    %     -Considering we want to create a probe file in the folder 
    %     'Sorting' that is on the Desktop, and we want to call this file 
    %     NewProbe.prb. This probe comes from the seventh group of 
    %     electrodes in 'elecGroup', which was calculated with a radius of 
    %     100um, and was originated by the 
    %     'recording' object:
    %
    %     mxw.util.generateProbe('~/Desktop/Sorting/NewProbe.prb', ...
    %       recording, elecGroup{7}, 100);
    %
    %

index = mod(find(fileManagerObj.rawMap.map.electrode == group'), length(fileManagerObj.rawMap.map.electrode));
index(index == 0) = length(fileManagerObj.rawMap.map.electrode);

xpos = fileManagerObj.rawMap.map.x(index)';
ypos = fileManagerObj.rawMap.map.y(index)';
channels = fileManagerObj.rawMap.map.channel(index)';

% Python expects 0-based channel index
channels = channels - 1; 

geometry = [channels', xpos', ypos'];

fid = fopen(fileName, 'w');
fprintf(fid, 'total_nb_channels = %d\n', length(channels));
fprintf(fid, 'radius = %f\n', radius);

fprintf(fid, 'channel_groups = {\n 1: {\n    ''channels'': [');

for iChannel = 1:length(channels) - 1
    fprintf(fid, '%d,', channels(iChannel));
end

fprintf(fid, '%d', channels(length(iChannel)));

fprintf(fid, '],\n');
fprintf(fid, '''graph'' : [],\n');
fprintf(fid, '''geometry'': {\n');

for iChannel = 1:size(geometry,1)
    fprintf(fid, '      %d: [%f,%f],\n', geometry(iChannel,1), geometry(iChannel,2), geometry(iChannel,3));
end

fprintf(fid, '}  }  }\n');
fclose(fid);
end
