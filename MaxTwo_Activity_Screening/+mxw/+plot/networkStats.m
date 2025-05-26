function networkStats( networkStatsStruct, varargin )
    % NETWORKSTATS plots the network activity statistics previously 
    % computed in 'mxw.networkActivity.computeNetworkStats'.
    % 
    % mxw.plot.networkStats(networkStatsStruct);
    %
    %   -The input parameters for this function are:
    %    -networkStatsStruct: struct containing the max amplitudes of the
    %                         peaks that crossed the threshold defined in
    %                         'mxw.networkActivity.computeNetworkStats' and 
    %                         the time differences between those amplitudes
    %    -varargin: ...
    %    -'Option': one of two options can be choosen:
    %       -'maxAmplitude': plots the distribution of peak maximum 
    %                        amplitudes
    %       -'maxAmplitudeTimeDiff': plots the distribution of time 
    %                                differences (in seconds) between the
    %                                maximum amplitudes of the peaks
    %    -'Figure': boolean that defines whether or not a new figure is 
    %               created for this plot. This is useful when the user 
    %               wants to show several plots in one figure by using the
    %               subplot option
    %    -'Bins': bin size used to plot the distribution
    %    -'Color'
    %    -'Axis': boolean that defines if the axis are plotted
    %    -'Xlimits': defines the minimum and maximum values on the x-axis
    %    -'Ylimits': defines the minimum and maximum values on the y-axis
    %    -'Title': title of the plot
    %    -'Ylabel': y-axis label
    %    -'Xlabel': x-axis label
    %                                   
    %  -Examples
    %     -Considering we want to plot the distribution of time differences 
    %     between the maximum peaks of activity contained in the structure 
    %     'netStats'. For this we will use a bin size of 50:
    %
    %     mxw.plot.networkStats(netStats, 'Option', ...
    %       'maxAmplitudeTimeDiff', 'Bins', 50);
    %
    %

p = inputParser;

p.addRequired('struct', @(x) isstruct(x));
p.addParameter('Option', []);
p.addParameter('Figure', true);
p.addParameter('Bins', 30);
p.addParameter('Color', 30);
p.addParameter('Axis', true);
p.addParameter('Xlimits', []);
p.addParameter('Ylimits', []);
p.addParameter('Title', 'Network Activity statistics');
p.addParameter('Ylabel', 'ylable');
p.addParameter('Xlabel', 'xlable');

p.parse(networkStatsStruct, varargin{:});
args = p.Results;

if isempty(args.Option)
    %errorMsg = sprintf(Please select a network statistics option to plot:\n *'maxAmplitude' \n *'maxAmplitudeTimeDiff'); 
    errorMsg = 'error';
    error(errorMsg);
    
elseif strcmp(args.Option, 'maxAmplitude')
    currentValue = networkStatsStruct.maxAmplitudesValues;
    
elseif strcmp(args.Option, 'maxAmplitudeTimeDiff')
    currentValue = networkStatsStruct.maxAmplitudeTimeDiff;

else
    %errorMsg = sprintf(Please select a network statistics option to plot:\n *'maxAmplitude' \n *'maxAmplitudeTimeDiff'); 
    errorMsg = 'error';
    error(errorMsg);
end

if args.Figure
    figure;
end

h = histogram(currentValue, args.Bins);
h.FaceColor = args.Color; h.EdgeColor = args.Color; h.FaceAlpha = 1;

if ~(args.Axis)
    axis off;
end

if ~isempty(args.Xlimits)
    xlim(args.Xlimits)
end

if ~isempty(args.Ylimits)
    ylim(args.Ylimits)
end

title(args.Title);
xlabel(args.Xlabel);
ylabel(args.Ylabel);
end