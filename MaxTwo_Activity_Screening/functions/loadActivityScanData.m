function activityScanData = loadActivityScanData(pathFileActivityScan)
%   Inputs:
%       pathFileActivityScan - A string specifying the full path to the
%                              activity scan data file.
%
%   Outputs:
%       activityScanData - A cell array where each cell contains the data
%                          for a well, loaded using mxw.fileManager.

    % Initialize activityScanData as an empty cell array. We'll pre-allocate
    % more efficiently as we load data, but starting empty is safer if
    % the number of wells is unknown.
    activityScanData = {};
    wellID = 1; % Start with the first well ID

    while true
        try
            % Attempt to load data for the current wellID
            activityScanData{wellID} = mxw.fileManager(pathFileActivityScan, wellID);
            wellID = wellID + 1; % Increment well ID for the next iteration

            if wellID == 25
                break;
            end

        catch ME
            % If an error occurs, it typically means there are no more wells
            % with the current wellID. Display the number of wells found.
            disp(['Number of Wells Found: ', num2str(wellID - 1)]);
            break; % Exit the loop
        end
    end
end
