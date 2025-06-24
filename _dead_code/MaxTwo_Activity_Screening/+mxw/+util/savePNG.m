function savePNG( directory, fileName, varargin )
    % SAVEPNG saves the latest figure as a .png image.
    % 
    % mxw.util.savePNG(directory, fileName);
    %
    %   -The input parameters for this function are:
    %    -directory: path to directory. It can be an already existing
    %                directory or a new one
    %    -fileName: name used to save the image
    %    -varargin: ...
    %    -'Resolution': resolution used to save the image
    %                                                
    %  -Examples
    %     -Considering we want to save the latest plotted figure that 
    %     contains a raster plot:
    %
    %     mxw.util.savePNG('/home/user/Desktop', 'this_is_a_rasterPlot');
    %
    %

p = inputParser;

p.addRequired('directory', @(x) ischar(x));
p.addRequired('fileName', @(x) ischar(x));
p.addParameter('Resolution', 1000);

p.parse(directory, fileName, varargin{:});
args = p.Results;

mkdir(directory);

if ~(strcmp(fileName(end-3:end), '.png'))
    fullPath = fullfile(directory, [fileName, '.png']);
else
    fullPath = fullfile(directory, fileName);
end

set(gcf, 'PaperPositionMode', 'auto');
print('-dpng', ['-r' num2str(args.Resolution)], fullPath);
end
