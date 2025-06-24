function screenActivity(dataPath, saveDir)
    
    addpath(genpath('functions'));

    fprintf('Screening Activity');
    activityScanData = loadActivityScanData(dataPath);


    % 2. Create and Save Plot
    fileName = "Activity Scan.png";
    plotWellActivityOverview(activityScanData, saveDir, fileName);

end