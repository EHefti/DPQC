function runNetworkPlots(doSave)
%RUNNETWORKPLOTS Summary of this function goes here
%   RUNNETWORKPLOTS('save') saves the figure into a user-specified file.
%
%   See also mxw.fileManager

%% Select file manually

[FileName,PathName,FilterIndex] = uigetfile('*.raw.h5');

pathToNetworkAnalysis = [PathName FileName];

%% Generate plots

% set threshold to detect bursts
% thr_burst = 1.1;
% gaussian_bin_size = 0.3;
% 
% % Load data information
% networkAnalysisFile = mxw.fileManager(pathToNetworkAnalysis);
% % Compute network activity
% networkAct = mxw.networkActivity.computeNetworkAct(networkAnalysisFile, 'BinSize', 0.02, 'file', 5,'GaussianSigma', gaussian_bin_size);
% networkStats = mxw.networkActivity.computeNetworkStats(networkAct, 'Threshold', thr_burst);
% % Plotting:
% figure('color', 'w','position',[0 100 1100 700]);
% % Raster Plot
% subplot(2, 2, 1);
% mxw.plot.rasterPlot(networkAnalysisFile, 'file', 6, 'Figure', false);box off;
% xlim([0 floor(max(networkAct.time))-0.5])
% % Histogram gaussian convolution
% subplot(2, 2, 3);
% mxw.plot.networkActivity(networkAct, 'Threshold', thr_burst, 'Figure', false);box off;
% hold on;plot(networkStats.maxAmplitudesTimes,networkStats.maxAmplitudesValues,'or')
% xlim([0 floor(max(networkAct.time))-0.5])
% ylim([0 3000])
% % Network statistics
% subplot(2, 2, 2);
% mxw.plot.networkStats(networkStats, 'Option', 'maxAmplitude', 'Figure', false, ...
% 'Ylabel', 'Counts', 'Xlabel', 'Burst peak [Hz]', 'Title', 'Burst Peak Distribution','Bins',5); box off;
% legend(['Mean Burst Peak = ',num2str(mean(networkStats.maxAmplitudesValues),'%.2f'), ' sd = ',num2str(std(networkStats.maxAmplitudesValues),'%.2f')])
% % Network statistics
% subplot(2, 2, 4);
% mxw.plot.networkStats(networkStats, 'Option', 'maxAmplitudeTimeDiff', 'Figure', false,...
% 'Ylabel', 'Counts', 'Xlabel', 'Interburst Interval [s]', 'Title', 'Interburst Interval Distribution','Bins',5); box off;
% legend(['Mean Interburst Interval = ',num2str(mean(networkStats.maxAmplitudeTimeDiff),'%.2f'),' sd = ',num2str(std(networkStats.maxAmplitudeTimeDiff),'%.2f')])
% 

%% Optional: save Figure


if exist('doSave') && strcmp(doSave,'save')
    [FileName,PathName,FilterIndex] = uiputfile('network.jpg','Save file name');
    saveas(gcf,[PathName FileName])
end
    

end

