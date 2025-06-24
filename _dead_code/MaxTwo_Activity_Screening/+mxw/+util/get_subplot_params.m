function [rows columns]=get_subplot_params(numSignals,varargin)

%GET_SUBPLOT_PARAMS  get number of rows and columns for subplot
%
%   [rows columns] = GET_SUBPLOT_PARAMS(numSignals)  numSignals is the amount of
%   subplots you want to generate
%
%   [rows columns] = GET_SUBPLOT_PARAMS(numSignals,'config',C) square
%   configuration for C = 'normal' (default), special "blocks" configuration to
%   visualize footprints in HiDens blocks
%
%   See also subplot

[config]=process_options(varargin,'config','normal')

if strcmp(config,'blocks')
    if numSignals==1
        rows = 1;
        columns = 1;
    else
        rows = floor(sqrt(numSignals/2));
        columns = ceil(sqrt(numSignals*2));
        while (rows * columns < numSignals)
            columns = columns + 1;
        end
    end
elseif strcmp(config,'superblocks')
    if numSignals==1
        rows = 1;
        columns = 1;
    elseif numSignals==2
        rows = 1;
        columns = 2;    
    else
        rows = floor(sqrt(numSignals/3));
        columns = ceil(sqrt(numSignals*3));
        while (rows * columns < numSignals)
            columns = columns + 1;
        end
    end
    
elseif strcmp(config,'normal')
    
    rows = floor(sqrt(numSignals));
    columns = ceil(sqrt(numSignals));
    while (rows * columns < numSignals)
        columns = columns + 1;
    end
    
end