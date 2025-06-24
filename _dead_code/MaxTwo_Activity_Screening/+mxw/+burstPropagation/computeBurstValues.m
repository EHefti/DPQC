function [burstVals out] = computeBurstValues(networkAnalysisFile,args)


%% 
thr_burst=args.thr_burst ; % in rms
binSize=args.binSize ;
gaussianBinSize=args.gaussianBinSize ; % in seconds
thr_start_stop=args.thr_start_stop ; % 0.3 means 30% value of the burst peak
prePeakTime=args.prePeakTime ; % in seconds
postPeakTime=args.postPeakTime ;
postPeakIgnore=args.postPeakIgnore ; % in seconds
numSpikesTiming=args.numSpikesTiming ;
numElsToAverage=args.numElsToAverage ;



%% Temporal burst propagation analysis

% Compute network activity
networkAct = mxw.networkActivity.computeNetworkAct(networkAnalysisFile, 'BinSize', binSize,'GaussianSigma', gaussianBinSize);
networkStats = mxw.networkActivity.computeNetworkStats(networkAct, 'Threshold', thr_burst);

relativeSpikeTimes = mxw.util.computeRelativeSpikeTimes(networkAnalysisFile);

ts = relativeSpikeTimes.time;
chs = relativeSpikeTimes.channel;

% get el-position info ready

chsConnected=networkAnalysisFile.rawMap.map.channel;
xpos = networkAnalysisFile.rawMap.map.x;
ypos = networkAnalysisFile.rawMap.map.y;
els = networkAnalysisFile.rawMap.map.electrode;


% STEP 1: for every peak, check if there is another one postPeakIgnore
% before. If there is one, discard the peak.

peakTs = networkStats.maxAmplitudesTimes;
indSelected = 1;
for i= 2:length(peakTs)
    
    if (peakTs(i)-peakTs(i-1))>postPeakIgnore
        indSelected(end+1) = i;
    end
end
peakSel = peakTs(indSelected);
ampSel = networkStats.maxAmplitudesValues(indSelected);

% STEP 2:

burstVals = cell(1,length(peakSel));
tic

clear xCenter yCenter numSpikesCon

burstToRemove = [];
for i= 1:length(peakSel)
    
    t_start =  peakSel(i)-prePeakTime;
    t_end = peakSel(i)+prePeakTime;
    
    
    
    indTmp = find(ts > t_start & ts < t_end);
    chsTmp = double(chs(indTmp));
    tsTmp = double(ts(indTmp));
    
    
    for ii=1:1024
        
        burstVals{i}.ts{ii}=tsTmp(find(chsTmp==ii)); % probably not needed later!
        burstVals{i}.numSpikes(ii)=length(find(chsTmp==ii));
        if ~isempty(burstVals{i}.ts{ii})
            len=length(burstVals{i}.ts{ii});
            burstVals{i}.startT(ii)=mean(burstVals{i}.ts{ii}(1:min(len,numSpikesTiming)));
        else 
            burstVals{i}.startT(ii)=0;
        end
        
    end
    
    [tf, loc] = ismember(chsTmp,chsConnected);
    
    % Extract the elements of a at those indexes.
    %     indexes = chsConnected(loc)
    
    x_pos= xpos(loc);
    y_pos= ypos(loc);
    elecs = double(els(loc));
    
    burstVals{i}.tsPerCh = [chsTmp tsTmp x_pos y_pos elecs];
    
    
    delVal=burstVals{i}.startT;

    delValCon = delVal(chsConnected);

    numSpikes = burstVals{i}.numSpikes(chsConnected);
    
    keep=find(delValCon>0);
    
    x2=xpos(keep);
    y2=ypos(keep);
    delValCon2 = delValCon(keep);
    numSpikes2 = numSpikes(keep);
    
    minVal=min(delValCon2);
    maxVal=max(delValCon2);
    
    delValCon2=delValCon2-minVal;
    
    % compute starting point:
    
    [delSorted, indSorting] = sort(delValCon2);
    
    minDel = mxw.util.percentile(delSorted,0.5);
    minInd=find(delSorted>minDel,1);
    
    try
        xCenter(i) = mean(x2(indSorting(minInd:(minInd+numElsToAverage))));
    
    yCenter(i) = mean(y2(indSorting(minInd:(minInd+numElsToAverage))));
    numSpikesCon(i)=sum(numSpikes2);
    
    burstVals{i}.xCenter = xCenter(i);
    burstVals{i}.yCenter = yCenter(i);
    burstVals{i}.numSpikesCon = numSpikesCon(i);
    burstVals{i}.numSpikes2 = numSpikes2;
    burstVals{i}.delValCon2 = delValCon2;
    burstVals{i}.x2=x2;
    burstVals{i}.y2=y2;
    burstVals{i}.xAll=xpos;
    burstVals{i}.yAll=ypos;
    burstVals{i}.elAll=elecs;
    
    
    catch
        burstToRemove(end+1) = i;
    end
end
toc

burstVals(burstToRemove)=[];
xCenter(burstToRemove)=[];
yCenter(burstToRemove)=[];
numSpikesCon(burstToRemove)=[];
peakSel(burstToRemove)=[];
ampSel(burstToRemove)=[];

% output params
out.xCenter = xCenter;
out.yCenter = yCenter;
out.numSpikesCon = numSpikesCon;
out.peakSel = peakSel;
out.ampSel = ampSel;
out.networkAct = networkAct;
out.networkStats = networkStats;
