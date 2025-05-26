function out = splitSpikesFiles(fManObj, splitTimes)

%% splitSpikesFiles(fileManagerObject)
%
%

fManObj_bkup = fManObj;


spikes = fManObj.fileObj(1).spikes;
ts=spikes.frameno-fManObj.fileObj(1).firstFrameNum;
Fs=fManObj.fileObj(1).samplingFreq;

splitTimes = double([splitTimes 0.1+ts(end)/Fs])

currentTime=0;

for i=1:length(splitTimes)
    
    currFManObj = mxw.fileManager(fManObj.referencePath);
    
    startTime = currentTime;
    endTime = splitTimes(i);

    currInds=find(ts>startTime*Fs & ts<=endTime*Fs);
    indsToRemove=find(ts<startTime*Fs | ts>=endTime*Fs);

    currFManObj.removeSpikes(indsToRemove);
    out{i}=currFManObj;
    
    currentTime = endTime;
    
end



