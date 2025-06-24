function rasterPlot( fileManagerORspikeTimes, varargin )
    % RASTERPLOT plots the spikes on every electrode as dots on time. The
    % spikes plotted are the ones detected by the recording software. 
    % 
    % mxw.plot.rasterPlot(fileManagerORspikeTimes);
    %
    %   -The input parameters for this function are:
    %    -fileManagerORspikeTimes: object of the class 'mxw.fileManager' or
    %                              precomputed spike times by using the 
    %                              function 
    %                              'mxw.util.computeRelativeSpikeTimes'
    %    -varargin: ...
    %    -'file': in case an object from the class 'mxw.fileManager' is
    %             set as the input and this object contains more than one 
    %             recording file, the file number to use has to be declared
    %    -'Figure': boolean that defines whether or not a new figure is 
    %               created for this plot. This is useful when the user 
    %               wants to show several plots in one figure by using the
    %               subplot option
    %    -'MarkerSize': size of the dots plotted as spikes
    %    -'Axis': boolean that defines if the axis are plotted
    %    -'Title': title of the plot
    %    -'Ylabel': y-axis label
    %    -'Xlabel': x-axis label
    %                                   
    %  -Examples
    %     -Considering we want the raster plot of file number 8 from an 
    %     object, 'networkRecording', of the 'mxw.fileManager' class: 
    %
    %     mxw.plot.rasterPlot(networkRecording, 'file', 8);
    %
    %     -Considering we want the raster plot from an object, 
    %     'networkRecording', of the 'mxw.fileManager' class that contains 
    %     only one file, and we also want to increase the size of the dots: 
    %
    %     mxw.plot.rasterPlot(networkRecording, 'MarkerSize', 4);
    %
    %
    
p = inputParser;

p.addRequired('struct', @(x) isobject(x) || isstruct(x));
p.addParameter('file', []);
p.addParameter('Figure', true);
p.addParameter('MarkerSize', 2);
p.addParameter('Axis', true);
p.addParameter('Title', 'Raster plot');
p.addParameter('Ylabel', 'Channels');
p.addParameter('Xlabel', 'Time (s)');

p.parse(fileManagerORspikeTimes, varargin{:});
args = p.Results;

if isa(fileManagerORspikeTimes, 'mxw.fileManager')
    relativeSpikeTimes = mxw.util.computeRelativeSpikeTimes(fileManagerORspikeTimes, 'file', args.file);

else
    relativeSpikeTimes = fileManagerORspikeTimes;
end

if args.Figure
    figure;
end

plot(relativeSpikeTimes.time, relativeSpikeTimes.channel, '.k', 'MarkerSize', args.MarkerSize)

if ~(args.Axis)
    axis off;
else

title(args.Title);
xlabel(args.Xlabel);
ylabel(args.Ylabel);
end