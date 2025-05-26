function waveforms( xpos, ypos, waveforms, varargin )
    % WAVEFORMS plots 'waveforms' as a map with xpos and ypos coordinates. 
    % 
    % mxw.plot.waveforms(xpos, ypos, waveforms);
    %
    %   -The input parameters for this function are:
    %    -xpos: 'x' coordinates of the electrodes used to get the
    %           'waveforms'
    %    -ypos: 'y' coordinates of the electrodes used to get the
    %           'waveforms'
    %    -waveforms: waveforms to be plotted
    %    -varargin: ...
    %    -'Figure': boolean that defines whether or not a new figure is 
    %               created for this plot. This is useful when the user 
    %               wants to show several plots in one figure by using the
    %               subplot option
    %    -'PlotFullArea': boolean that defines if the whole MEA area is
    %                     plotted, or just the area of interest
    %    -'XScale': scale that modifies the size of the waveforms on the
    %               x-axis
    %    -'YScale': scale that modifies the size of the waveforms on the
    %               y-axis
    %    -'WaveformColor': waveforms color
    %    -'WaveformWidth': waveforms trace width
    %    -'Title': title of the plot
    %                                   
    %  -Examples
    %     -Considering now we want to plot the waveforms contained in 
    %     'waveforms' which are mapped to the coordinates 'xcoord' and 
    %     'ycoord'. We want these waveforms to be blue and thicker than 
    %     the default value:
    %
    %     mxw.plot.waveforms(xcoord, ycoord, waveforms, ...
    %       'WaveformColor', 'blue', 'WaveformWidth', 3);
    %
    %

p = inputParser;

p.addRequired('xpos');
p.addRequired('ypos');
p.addRequired('waveforms');

p.addParameter('Figure', true);
p.addParameter('PlotFullArea', true);
p.addParameter('XScale', 1);
p.addParameter('YScale', 1);
p.addParameter('WaveformColor', 'black');
p.addParameter('WaveformWidth', 1);
p.addParameter('Title', 'title');

p.parse(xpos, ypos, waveforms, varargin{:});
args = p.Results;

if args.Figure
    figure;
end

hold on;

xBin = 1/(size(waveforms,1));
yBin = 1/abs(min(waveforms(:)));

minDistX = min(abs(diff(xpos)));
minDistY = max(abs(diff(ypos)));

scaledWaveformX = ((-size(waveforms,1)/2) : ((size(waveforms,1)/2)-1)) * xBin * 0.7*minDistX * args.XScale;
scaledWaveformY = -1 * waveforms * yBin * 0.5*minDistY * args.YScale;

for iWaveform = 1:length(waveforms)
    line (xpos(iWaveform) + scaledWaveformX, ypos(iWaveform) + scaledWaveformY(:, iWaveform),...
        'LineWidth', args.WaveformWidth, 'Color', args.WaveformColor);
%     plot(xpos(iWaveform) + scaledWaveformX, ypos(iWaveform) + scaledWaveformY(:, iWaveform),...
%         'LineWidth', args.WaveformWidth, 'Color', args.WaveformColor);
    
end

hold off;

axis ij;
axis equal;

if args.PlotFullArea
    xlim([165 4010]);
    ylim([155 2250]);
end

title(args.Title, 'fontsize', 12);
end
