function plotWellActivityOverview(activityScanData, pathRecording, gridFilename)
%   Inputs:
%       activityScanData - A cell array containing data for each well,
%                          typically loaded using mxw.fileManager.
%       pathRecording    - A string specifying the directory where the
%                          output plot PNG file should be saved.
%   Outputs:
%       None - The .png is saved in the chosen folder

    % Validate inputs
    if nargin < 2
        error('Both activityScanData and pathRecording are required inputs.');
    end
    
    if ~iscell(activityScanData)
        error('activityScanData must be a cell array.');
    end
    
    if ~ischar(pathRecording) || ~isfolder(pathRecording)
        error('pathRecording must be a string specifying a valid directory.');
    end
    
    thr_amp = 10;
    numWells = length(activityScanData);
    
    % Only process up to 24 wells for the 8x6 grid
    if numWells > 24
        fprintf('Warning: Processing only the first 24 wells for the 8x6 grid plot.\n');
        numWells = 24;
    elseif numWells < 1
        fprintf('No wells found in activityScanData to plot.\n');
        return;
    end
    
    fprintf('Analyzing %d wells for 8x6 grid plot (Wells 1-%d, Map above Hist)...\n', numWells, numWells);
    
    % --- Create the main figure ---
    hFig = figure('Name', 'All Wells Analysis - 8x6 Grid (Map/Hist Stacked)', 'color', 'w', ...
        'WindowState', 'maximized', 'Position', [100, 100, 1200, 800]);
    gridRows = 8; % 4 well-rows * 2 plot-rows/well-row
    gridCols = 6; % 6 wells per row horizontally
    
    for wellIndex = 1:numWells % wellIndex represents well number 1-24
    
        currentWellData = activityScanData{wellIndex};
    
        % Calculate subplot indices for this well (map above hist)
        wellRowGroup = ceil(wellIndex / gridCols); % Which 'well row' (1-4)
        wellCol = mod(wellIndex - 1, gridCols) + 1; % Which column (1-6)
    
        mapRowIndex = (wellRowGroup - 1) * 2 + 1;  % Actual row index for Map (1, 3, 5, 7)
        histRowIndex = mapRowIndex + 1;            % Actual row index for Hist (2, 4, 6, 8)
    
        mapSubplotIndex = (mapRowIndex - 1) * gridCols + wellCol;
        histSubplotIndex = (histRowIndex - 1) * gridCols + wellCol;
    
        fprintf('Processing Well %d -> Map Subplot %d, Hist Subplot %d\n', wellIndex, mapSubplotIndex, histSubplotIndex);
    
        % --- Check Data ---
        if isempty(currentWellData)
            subplot(gridRows, gridCols, mapSubplotIndex); title(sprintf('W%d No Data', wellIndex)); axis off;
            subplot(gridRows, gridCols, histSubplotIndex); title(sprintf('W%d No Data', wellIndex)); axis off;
            continue;
        end
    
        % --- Analyze and Plot (within try-catch) ---
        try
            % 1. Get the 90th percentile spike amplitude value for each electrode
            amplitude90perc = abs(mxw.activityMap.computeAmplitude90percentile(currentWellData));
    
            if isempty(amplitude90perc) || all(isnan(amplitude90perc(:)))
                subplot(gridRows, gridCols, mapSubplotIndex); title(sprintf('W%d No FR', wellIndex)); axis off;
                subplot(gridRows, gridCols, histSubplotIndex); title(sprintf('W%d No FR', wellIndex)); axis off;
                continue;
            end
    
            % 2. Determine Max Firing Rate
            nonZeroAmplitude = amplitude90perc(amplitude90perc > 0);
            if isempty(nonZeroAmplitude)
                max_amp = 1.0;
            else
                max_amp = mxw.util.percentile(amplitude90perc(amplitude90perc~=0),99);
                if isnan(max_amp) || isinf(max_amp) || max_amp <= 0, max_amp = max([1.0; nonZeroAmplitude]); end
            end
    
            % --- Plotting in Grid ---
            % 3. Plot Activity Map (Top)
            subplot(gridRows, gridCols, mapSubplotIndex);
            mxw.plot.activityMap(currentWellData, amplitude90perc, 'Ylabel', '[\muV]',...
                'CaxisLim', [10 max_amp], 'Figure',false, 'Title', sprintf('Well %d Map', wellIndex));
            axis off;
            title( sprintf('Well %d Map', wellIndex), 'FontSize', 8 );
    
            % 4. Plot Firing Rate Histogram (Bottom)
            subplot(gridRows, gridCols, histSubplotIndex);
            histogram(amplitude90perc(amplitude90perc>thr_amp),ceil(0:1:max_amp))
            title_str = sprintf('Well %d Hist', wellIndex);
            xlim_hist = [thr_amp, max(thr_amp + 1, ceil(max_amp))];
    
            title(title_str, "FontSize",8);
            xlim(xlim_hist);
            ylabel('Counts');
            xlabel('Spike Amplitude [\muV]');
            box off;
    
        catch ME
            fprintf('Error processing Well %d: %s\n', wellIndex, ME.message);
            subplot(gridRows, gridCols, mapSubplotIndex); title(sprintf('W%d Plot Error', wellIndex)); axis off;
            subplot(gridRows, gridCols, histSubplotIndex); title(sprintf('W%d Plot Error', wellIndex)); axis off;
            continue;
        end
    end % End well loop
    
    fprintf('--- Grid plot generation complete. Preparing to save with upscaling... ---\n');
    
    % --- Save the entire figure once after the loop with specific size and resolution ---
    try
        %gridFilename = 'WellActivityAmp.png'; % Descriptive filename
        fullGridSavePath = fullfile(pathRecording, gridFilename);
    
        fprintf('Attempting to save upscaled figure to: %s\n', fullGridSavePath);
    
        % --- Settings for Upscaling ---
        targetWidth_cm = 50;
        targetHeight_cm = 34;
        resolution_dpi = 300; % Dots Per Inch (Increase to 300 or 600 for higher quality)
    
        % Set paper units and size
        set(hFig, 'PaperUnits', 'centimeters');
        set(hFig, 'PaperSize', [targetWidth_cm, targetHeight_cm]);
    
        % Set paper position manually to use most of the paper size
        % [left margin, bottom margin, width on paper, height on paper]
        set(hFig, 'PaperPositionMode', 'manual');
        set(hFig, 'PaperPosition', [0.2, 0.2, targetWidth_cm - 2, targetHeight_cm - 2]);
    
        % Prepare resolution argument for print command
        resolution_opt = sprintf('-r%d', resolution_dpi);
    
        drawnow; % Ensure figure updates visually before printing to file
    
        % Use print command for better control over size and resolution
        % '-dpng' specifies PNG format
        print(hFig, '-dpng', resolution_opt, fullGridSavePath);
    
        fprintf('Successfully saved upscaled figure: %s\n', fullGridSavePath);
    
    catch ME_save
        fprintf('*** ERROR saving the upscaled figure: %s\n', ME_save.message);
        fprintf('Figure not saved. It might still be open.\n');
    end

end
