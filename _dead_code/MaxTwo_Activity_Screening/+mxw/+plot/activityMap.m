function activityMap( fileManagerObj, value, varargin )
    % ACTIVITYMAP plots the 'value' vector as a map, given the coordinates 
    % on 'fileManagerObj.processedMap'. The map represents the whole MEA or
    % an area within it.
    % 
    % mxw.plot.activityMap(fileManagerObj, value)
    %
    %   -The input parameters for this function are:
    %    -fileManagerObj: object of the class 'mxw.fileManager'
    %    -value: vector containing the values to plot in the map. This
    %            vector can be computed by any of the functions in
    %            'mxw.activityMap...'
    %    -varargin: ...
    %    -'Figure': boolean that defines whether or not a new figure is 
    %               created for this plot. This is useful when the user 
    %               wants to show several plots in one figure by using the
    %               subplot option
    %    -'ColorMap': defines the map's color used in the plot, e.g. 
    %                 parula, hot, grey
    %    -'RevertColorMap': boolean that reverts the color map scale
    %    -'CaxisLim': defines the minimum and maximum values of the color 
    %                 scale used on the map   
    %    -'Interpolate': boolean that defines whether the values in the map
    %                    are interpolated or not. When the map has
    %                    consecutive coordinates (such as a map of the 
    %                    whole MEA, or one high dense block of electrodes),
    %                    'Interpolate' provides a good picture of the
    %                    activity. However, when the map is sparse, or
    %                    contains blocks of electrodes separted from each
    %                    other, 'Interpolate' provides a wrong map. When 
    %                    'Interpolate' is set to false, ACTIVITYMAP plots a 
    %                    scatter plot
    %    -'PlotFullArea': boolean that defines if the whole MEA area is
    %                     plotted, or just the area of interest
    %    -'PointSize': Defines the size of the points in the scatter plot
    %                  when no interpolation is used
    %    -'Ylabel': label of the color bar
    %    -'Title': title of the activity map
    %                                   
    %  -Examples
    %     -Considering we want to plot the activity map based on the spike 
    %     count previously computed on 'mxw.activityMap.computeSpikeCount'.
    %     This spike count is computed over the whole MEA, therefore we can
    %     keep the 'Interpolate' with its default value (true). Moreover,
    %     we want to change the color's map and the limits of the color
    %     bar:
    %
    %     mxw.plot.activityMap(spikeCount, 'ColorMap', 'hot', ...
    %       'CaxisLim', [0 200], 'Ylabel', 'number of spikes');
    %
    %     -Considering we want to plot the activity map based on the mean 
    %     amplitude of the spikes of two high dense blocks previously
    %     computed on 'mxw.activityMap.computeMeanAmplitude'. Now we don't
    %     want to interpolate but use a scatter plot, we will also keep the
    %     rest inputs with thier default values:
    %
    %     mxw.plot.activityMap(spikeMean, 'Interpolate', false);
    %
    %     -Considering we want to plot three activity maps on the same 
    %     figure:
    %      
    %     figure
    %     subplot(3,1,1)
    %     mxw.plot.activityMap(spikeCount, 'Figure', false, 'CaxisLim', ...
    %       [0 100], 'Ylabel', 'number of spikes');
    %     subplot(3,1,2)
    %     mxw.plot.activityMap(spikeRate, 'Figure', false, 'CaxisLim', ...
    %       [0 2], 'Ylabel', 'frequency');
    %     subplot(3,1,3)
    %     mxw.plot.activityMap(spikeMean, 'Figure', false, 'CaxisLim', ...
    %       [-50 0], 'Ylabel', 'amplitude');
    %
    %
try
axisMin = mxw.util.percentile(value, 10);
axisMax = mxw.util.percentile(value, 100);
catch
    axisMin = 0;
    axisMax = 1;
end

p = inputParser;

p.addRequired('obj', @(x) isobject(x));
p.addRequired('value', @(x) isvector(x));
p.addParameter('Figure', true);
p.addParameter('ColorMap', parula);
p.addParameter('RevertColorMap', false);
p.addParameter('CaxisLim', [axisMin axisMax]);
p.addParameter('Interpolate', true);
p.addParameter('PlotFullArea', true);
p.addParameter('LimToActive', true);
p.addParameter('PointSize', 100);
p.addParameter('Ylabel', 'ylabel');
p.addParameter('Title', 'Activity Map');

p.parse(fileManagerObj, value, varargin{:});
args = p.Results;

if args.Figure
    figure;
end

if args.Interpolate
    F = scatteredInterpolant(args.obj.processedMap.xpos, args.obj.processedMap.ypos, args.value, 'natural');
    [x, y] = meshgrid(unique(args.obj.processedMap.xpos), unique(args.obj.processedMap.ypos));
    qz = F(x, y);
    
    imagesc([min(args.obj.processedMap.xpos), max(args.obj.processedMap.xpos)], [min(args.obj.processedMap.ypos), max(args.obj.processedMap.ypos)], qz)
    
else
    
    scatter(args.obj.processedMap.xpos, args.obj.processedMap.ypos, args.PointSize, args.value, 'filled', 's');
end

axis ij;
% axis equal;

colormap(args.ColorMap);
c = colorbar;

if args.RevertColorMap
    colormap(flipud(args.ColorMap));
end

if args.PlotFullArea
    xlim([165 4010]);
    ylim([155 2250]);
end
if args.LimToActive
    
    xlim([min(args.obj.processedMap.xpos)-40 max(args.obj.processedMap.xpos+40)]);
    ylim([min(args.obj.processedMap.ypos)-40 max(args.obj.processedMap.ypos+40)]);
end
axis off;
box off;

caxis(args.CaxisLim);
ylabel(c, args.Ylabel);

title(args.Title, 'fontsize', 12);
end