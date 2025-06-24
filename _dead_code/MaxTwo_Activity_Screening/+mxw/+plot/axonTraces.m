function axonTraces( xpos, ypos, waveforms, varargin )
    % AXONTRACES plots the axonal traces previously computed in
    % 'mxw.axonalTracking.computeAxonTraces'. Three different types of 
    % plots can be choosen here: one plotting a heat map with the largest
    % values from 'waveforms', one plotting the area defined by 'xpos' and 
    % 'ypos' and a circle indicating where is the largest value from 
    % 'waveform', and one plotting all the 'waveforms' values.  
    % 
    % mxw.plot.axonTraces(xpos, ypos, waveforms);
    %
    %   -The input parameters for this function are:
    %    -xpos: 'x' coordinates of the electrodes used to get the
    %           'waveforms'
    %    -ypos: 'y' coordinates of the electrodes used to get the
    %           'waveforms'
    %    -waveforms: waveforms of one of the group of electrodes computed
    %                on 'mxw.axonalTracking.computeAxonTraces'
    %    -varargin: ...
    %    -'Figure': boolean that defines whether or not a new figure is 
    %               created for this plot. This is useful when the user 
    %               wants to show several plots in one figure by using the
    %               subplot option
    %    -'PlotFullArea': boolean that defines if the whole MEA area is
    %                     plotted, or just the area of interest
    %    -'PlotHeatMap': boolean that defines whether the heat map is 
    %                    plotted or not. The heat map uses only the largest
    %                    value of each waveform in 'waveforms' and the plot
    %                    is similar to an activity map. The heat map is a
    %                    scatter plot by default
    %    -'PointSize': Defines the size of the points in the scatter plot
    %    -'Interpolate': boolean that defines if the values for the heat 
    %                    map are interpolated. Use this just when the
    %                    'xpos' and 'ypos' coordinates are consecutives
    %    -'ColorMap': defines the map's color used in the plot, e.g. 
    %                 parula, hot, grey
    %    -'RevertColorMap': boolean that reverts the color map scale
    %    -'CaxisLim': defines the minimum and maximum values of the color 
    %                 scale used on the map   
    %    -'PlotCircle': boolean that defines if the circle indicating the
    %                   largest value from 'waveforms' is plotted on the 
    %                   map or not
    %    -'CircleWidth': circle line width
    %    -'CircleColor': circle line color
    %    -'HeatMapFeature': 'pkpkAmp' / 'minAmp' / 'latency'
    %    -'PlotWaveforms': boolean that defines whether the waveforms in
    %                      'wafevorms' are plotted or not. This option uses
    %                      the function 'mxw.plot.waveforms'
    %    -'XScale': scale that modifies the size of the waveforms on the
    %               x-axis
    %    -'YScale': scale that modifies the size of the waveforms on the
    %               y-axis
    %    -'WaveformColor': waveforms color
    %    -'WaveformWidth': waveforms trace width
    %    -'Title': title of the plot
    %    -'Ylabel': label of the color bar
    %                                   
    %  -Examples
    %     -Considering we want to plot the axon traces heat map of the 
    %     electrode group number 5 in the struct 'axonalTraces' obtained by 
    %     using the function 'mxw.axonalTracking.computeAxonTraces'. Here 
    %     we are also plotting the circle indicating the largest value 
    %     among the waveforms: 
    %     
    %     mxw.plot.axonTraces(axonalTraces.map.x, axonalTraces.map.y, ...
    %       axonalTraces.traces{5}, 'Title', 'Axon Tracking', ...
    %       'Ylabel', 'Amplitude');
    %
    %     -Considering we want the same plot as before but now 
    %     interpolating the values. This interpolation is doable and
    %     presents reasonable results since we are considering that the
    %     'axonalTraces' structure is over the whole MEA. This time we do 
    %     not want the circle indicating the largest value among the 
    %     waveforms:
    %
    %     mxw.plot.axonTraces(axonalTraces.map.x, axonalTraces.map.y, ...
    %       axonalTraces.traces{5}, 'Interpolate', true, 'PlotCircle', ...
    %       false, 'Title', 'Axon Tracking', 'Ylabel', 'Amplitude');
    %
    %     -Considering now we want to plot the tenth group in 
    %     'axonalTraces', furthermore we want to only plot the waveforms 
    %     and nothing else:
    %
    %     mxw.plot.axonTraces(axonalTraces.map.x, axonalTraces.map.y, ...
    %       axonalTraces.traces{5}, 'PlotHeatMap', false, 'PlotCircle', ...
    %       false, 'PlotWaveforms', true, 'Title', 'Axon Waveforms');
    %
    %     -Considering we want to plot the three types of plots for the 
    %     first group in 'axonalTraces', each one in a different subplot:
    %
    %     figure
    %     subplot(3,1,1)
    %     mxw.plot.axonTraces(axonalTraces.map.x, axonalTraces.map.y, ...
    %       axonalTraces.traces{1}, 'Figure', false, 'PlotCircle', ...
    %       false, 'Title', 'Axon Tracking', 'Ylabel', 'Amplitude');
    %     subplot(3,1,2)
    %       mxw.plot.axonTraces(axonalTraces.map.x, axonalTraces.map.y, ...
    %       axonalTraces.traces{1}, 'Figure', false, 'PlotHeatMap', ...
    %       false);
    %     subplot(3,1,3)
    %       mxw.plot.axonTraces(axonalTraces.map.x, axonalTraces.map.y, ...
    %       axonalTraces.traces{1}, 'Figure', false, 'PlotHeatMap', ...
    %       false, 'PlotCircle', false, 'PlotWaveforms', true);
    %
    %


if ~isrow(xpos)
    xpos = xpos';
end

if ~isrow(ypos)
    ypos = ypos';
end

if (size(waveforms, 2) ~= size(xpos, 2))
    waveforms = waveforms';
end

% axisMin = mxw.util.percentile(min(waveforms), 1);
% axisMax = mxw.util.percentile(min(waveforms), 99);

axisMin = min(min(waveforms));
axisMax = 0;

p = inputParser;

p.addParameter('Figure', true);
p.addParameter('PlotFullArea', false);

p.addParameter('PlotHeatMap', true);
p.addParameter('HeatMapFeature', 'pkpkAmp');



p.addParameter('PointSize', 100);
p.addParameter('Interpolate', false);
p.addParameter('ColorMap', parula);
p.addParameter('RevertColorMap', false);
p.addParameter('CaxisLim', [axisMin axisMax]);

% p.addParameter('PlotContours', true);

p.addParameter('PlotWaveforms', false);
p.addParameter('NormalizeByLocalMax', false);
p.addParameter('WaveformSize', 16);
p.addParameter('WaveformColor', [0.5 0.5 0.5]);
p.addParameter('WaveformWidth', 1);
p.addParameter('uVolts2uMetersScale', 10);
p.addParameter('Title', 'title');
p.addParameter('Ylabel', 'ylabel');
p.addParameter('Border', 10);

p.parse(varargin{:});
args = p.Results;

if args.Figure
    figure('color','w');
end

hold on;

if strcmp(args.HeatMapFeature,'pkpkAmp')
    featureValues = max(waveforms)-min(waveforms);
elseif strcmp(args.HeatMapFeature,'minAmp')
    featureValues = min(waveforms)';
elseif strcmp(args.HeatMapFeature,'latency')
    
    % not really working.
    [va, inn] = min(waveforms);
    [va2, first_el] = min(va);
    
    lat = (inn-inn(first_el))/20/4;
    featureValues = lat;
end
    

if args.PlotHeatMap
    if args.Interpolate
        F = scatteredInterpolant(xpos', ypos', featureValues, 'nearest');
        [x, y] = meshgrid(unique(xpos), unique(ypos));
        qz = F(x, y);
        
        imagesc([min(xpos), max(xpos)], [min(ypos), max(ypos)], qz)
        
    else
        
        scatter(xpos, ypos, args.PointSize, featureValues, 'filled', 's');
    end
    
    colormap(args.ColorMap);
    
%     if args.RevertColorMap
        colormap(flipud(args.ColorMap));
%     end
    
    c = colorbar;
%      caxis(args.CaxisLim);
    ylabel(c, args.Ylabel);
end

% if args.PlotContours
%     contour(xpos, ypos, min(waveforms), [min(min(waveforms)) * 0.5, min(min(waveforms)) * 0.5], 'k', 'linewidth', 1);
% end

if args.PlotWaveforms
    normalizedWaveforms = waveforms ./ max(max(abs(waveforms)));
    
    if args.NormalizeByLocalMax
        normalizedWaveforms = waveforms ./ repmat(max(abs(waveforms)), size(waveforms, 1), 1);
    end
    
    step = args.WaveformSize / size(normalizedWaveforms, 1);
    left = (xpos - args.WaveformSize/2)';
    
    waveforms2Plot = (-normalizedWaveforms * args.uVolts2uMetersScale) + repmat(ypos, size(normalizedWaveforms,1), 1);
    
    tempStep = cumsum(repmat(step, size(normalizedWaveforms, 1) + 1, size(normalizedWaveforms, 2)), 1);
    left2Rigth = repmat(left', size(normalizedWaveforms, 1), 1) + tempStep(1:end-1, :);
    
    plot(left2Rigth, waveforms2Plot, 'color', args.WaveformColor, 'Linewidth', args.WaveformWidth)
end

hold off;

axis ij;
axis equal;

if args.PlotFullArea
    xlim([165 4010]);
    ylim([155 2250]);
end

xmin=min(xpos);
xmax=max(xpos);
ymin=min(ypos);
ymax=max(ypos);

bord=args.Border;
xlim([xmin-bord xmax+bord])
ylim([ymin-bord ymax+bord])


title(args.Title, 'fontsize', 12);
end

