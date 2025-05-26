function electrodeSelection( fileManagerObj, map, selectedElectrodes,  varargin )
    % ELECTRODESELECTION plots the electrodes in 'selectedElectrodes' on
    % top of the activity map defined by 'map'.
    %
    % mxw.plot.electrodeSelection(fileManagerObj, map, selectedElectrodes)
    %
    %   -The input parameters for this function are:
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    -map: vector containing the values of a map. This vector comes 
    %          from any of the activity maps computed by the functions in
    %          'mxw.activityMap...'
    %    -'selectedElectrodes': electrodes to be plotted on top of the
    %                           activity map. These electrodes can be 
    %                           computed with any of the functions in
    %                           'mxw.util.electrodeSelection...'
    %    -varargin: See the varargin from 'mxw.plot.activityMap'
    %
    %  -Examples
    %     -Considering we want to plot the electrodes in 
    %     'networkRecElectrodes' coming from the function 
    %     'mxw.util.electrodeSelection.networkRec(fileManagerObj)', on ...
    %     top of the spike count activity map:
    %
    %     mxw.plot.electrodeSelection(fileManagerObj, spikeCount, ...
    %       networkRecElectrodes, 'Ylabel', 'Spike count', 'Title', ...
    %       'Electrodes for network recording');
    %
    %

electrodes = selectedElectrodes.electrodes;

mxw.plot.activityMap(fileManagerObj, map, varargin{:});
hold
plot(fileManagerObj.processedMap.xpos(electrodes), fileManagerObj.processedMap.ypos(electrodes),'ro');
end