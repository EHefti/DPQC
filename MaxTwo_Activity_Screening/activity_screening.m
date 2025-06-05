% --- Plot Amplitude Overview ---

function screenActivity(dataPath, saveDir)

    fprintf('Screening Activity');
    activityScanData = loadActivityScanData(dataPath);


    % 2. Create and Save Plot
    fileName = "Activity Scan.png"
    plotWellActivityOverview(activityScanData, saveDir, fileName);

end


cd (fullfile(matlabroot,'extern','engines','python'))
system('python setup.py install')