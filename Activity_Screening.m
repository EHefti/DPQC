% --- Plot Amplitude Overview ---
% Functions are stored in MaxTwo_Activity_Screening folder
clear
addpath(genpath("MaxTwo_Activity_Screening"))

% Set paths
dataPath = 'S:/group/hierlemann02/intermediate_data/Maxtwo/lkaupp/Dup15q/250506/T003104/ActivityScan/000009/data.raw.h5';
saveDir = 'S:/group/hierlemann02/intermediate_data/Maxtwo/lkaupp/Dup15q/250506/';
mkdir(saveDir)

% Process data
screenActivity(dataPath, saveDir)