function [beatValues valueMatrix] = computeBeatValues(cardiacFile, deltaSamples, minNumSpikes)


%% Parameters

numElsToAverage=4;



%% detect Beats

relativeSpikeTimes = mxw.util.computeRelativeSpikeTimes(cardiacFile);
ts = relativeSpikeTimes.time;
tsSamples = cardiacFile.fileObj.samplingFreq*ts;

% This function detects the beats and groups them i
gr = mxw.cardiacAnalysis.detectBeats(tsSamples, deltaSamples, minNumSpikes);


%% prepare all required info

chs = relativeSpikeTimes.channel;

% get el-position info ready

chsConnected=cardiacFile.rawMap.map.channel;
xpos = cardiacFile.rawMap.map.x;
ypos = cardiacFile.rawMap.map.y;
els = cardiacFile.rawMap.map.electrode;


%%

beatValues = cell(1,length(gr));

clear xCenter yCenter


for i= 1:length(gr)
    
    beatValues{i}.tStart=ts(gr{i}.iStart);
    beatValues{i}.tEnd=ts(gr{i}.iEnd);
    beatValues{i}.duration = ts(gr{i}.iEnd)-ts(gr{i}.iStart);
    
    if i==1
        beatValues{i}.tFromLast=NaN;
    else
        beatValues{i}.tFromLast=beatValues{i}.tStart-beatValues{i-1}.tStart;
    end
    beatValues{i}.freq = 1/beatValues{i}.tFromLast;
    % extract spikes here:
    
    ind = gr{i}.ind;
    chsTmp = double(chs(ind));
    tsTmp = double(ts(ind));
    
    [tf, loc] = ismember(chsTmp,chsConnected);
    
    % Extract the elements of a at those indexes.
    
    x_pos= xpos(loc);
    y_pos= ypos(loc);
    elecs = double(els(loc));
    
    beatValues{i}.beatCoverage = 100*length(unique(elecs))/length(chsConnected);
    
    beatValues{i}.tsPerCh = [chsTmp tsTmp x_pos y_pos elecs];
    
    % Beat initiation point
    
    xInit = mean(x_pos(1:numElsToAverage));
    yInit = mean(y_pos(1:numElsToAverage));
    
    beatValues{i}.xInit = xInit;
    beatValues{i}.yInit = yInit;
    
    % velocity per beat
    
    % Compute distances between burst center and all els
    A =  beatValues{i}.tsPerCh;
    
    [C,ia,ic] = unique(A(:,5),'stable');
    d = mxw.util.pdist2([xInit yInit],[x_pos(ia) y_pos(ia)],'euclidean');
    
    deltaT =1000*(tsTmp(ia)-tsTmp(1)); % in ms
    
    x = deltaT;
    y = d';
    
    % include intercept
    X = [ones(length(x),1) x];

    %   b=X\y; % regression coefficioent (velocity)
    %   yCalc1 = b(1)+b(2)*x; % slope
    
    b1=x\y; % regression coefficient (velocity)

    yCalc1 = b1*x; % slope
    Rsq1 = 1 - sum((y - yCalc1).^2)/sum((y - mean(y)).^2);

    beatValues{i}.vel = round(b1/1000,4);
    beatValues{i}.Rsq1 = round(Rsq1,3);
    
% velocity plot
% figure('color','w')
% scatter(x,y,'o')
% xlabel('Latency [ms]')
% ylabel('Distance [\mum]')
% hold on
% plot(x,yCalc1,'r')
% title({['Velocity = ' num2str(round(b1/1000,2)) 'm/s'], ['R = ' num2str(round(Rsq1,2))]})
% box on
% 
% 
%     figure
%     scatter(x_pos(ia), y_pos(ia),20,deltaT,'filled')%,'MarkerEdgeColor','k')
%     axis ij
    beatValues{i}.x_pos=x_pos(ia);
    beatValues{i}.y_pos=y_pos(ia);
    beatValues{i}.deltaT=deltaT;
    
    valueMatrix(i,:) = [beatValues{i}.tStart beatValues{i}.tEnd beatValues{i}.duration ...
                        beatValues{i}.tFromLast beatValues{i}.freq ...
                        beatValues{i}.xInit beatValues{i}.yInit ...
                        beatValues{i}.vel beatValues{i}.Rsq1 ...
                        beatValues{i}.beatCoverage];
end



% %%
%
% peakSel = peakTs(indSelected);
% ampSel = networkStats.maxAmplitudesValues(indSelected);
%
% % STEP 2:
%
% burstVals = cell(1,length(peakSel));
% tic
%
% clear xCenter yCenter numSpikesCon
%
% burstToRemove = [];
% for i= 1:length(peakSel)
%
%     t_start =  peakSel(i)-prePeakTime;
%     t_end = peakSel(i)+prePeakTime;
%
%
%     indTmp = find(ts > t_start & ts < t_end);
%     chsTmp = double(chs(indTmp));
%     tsTmp = double(ts(indTmp));
%
%
%     [tf, loc] = ismember(chsTmp,chsConnected);
%
%     % Extract the elements of a at those indexes.
%     %     indexes = chsConnected(loc)
%
%     x_pos= xpos(loc);
%     y_pos= ypos(loc);
%     elecs = double(els(loc));
%
%
%     for ii=1:1024
%
%         burstVals{i}.ts{ii}=tsTmp(find(chsTmp==ii)); % probably not needed later!
%         burstVals{i}.numSpikes(ii)=length(find(chsTmp==ii));
%         if ~isempty(burstVals{i}.ts{ii})
%             len=length(burstVals{i}.ts{ii});
%             burstVals{i}.startT(ii)=mean(burstVals{i}.ts{ii}(1:min(len,numSpikesTiming)));
%         else
%             burstVals{i}.startT(ii)=0;
%         end
%
%     end
%
%
%     delVal=burstVals{i}.startT;
%
%     delValCon = delVal(chsConnected);
%
%     numSpikes = burstVals{i}.numSpikes(chsConnected);
%
%     keep=find(delValCon>0);
%
%     x2=xpos(keep);
%     y2=ypos(keep);
%     delValCon2 = delValCon(keep);
%     numSpikes2 = numSpikes(keep);
%
%     minVal=min(delValCon2);
%     maxVal=max(delValCon2);
%
%     delValCon2=delValCon2-minVal;
%
%     % compute starting point:
%
%     [delSorted, indSorting] = sort(delValCon2);
%
%     minDel = mxw.util.percentile(delSorted,0.5);
%     minInd=find(delSorted>minDel,1);
%
%     try
%         xCenter(i) = mean(x2(indSorting(minInd:(minInd+numElsToAverage))));
%
%     yCenter(i) = mean(y2(indSorting(minInd:(minInd+numElsToAverage))));
%     numSpikesCon(i)=sum(numSpikes2);
%
%     burstVals{i}.xCenter = xCenter(i);
%     burstVals{i}.yCenter = yCenter(i);
%     burstVals{i}.numSpikesCon = numSpikesCon(i);
%     burstVals{i}.numSpikes2 = numSpikes2;
%     burstVals{i}.delValCon2 = delValCon2;
%     burstVals{i}.x2=x2;
%     burstVals{i}.y2=y2;
%     burstVals{i}.xAll=xpos;
%     burstVals{i}.yAll=ypos;
%     burstVals{i}.elAll=elecs;
%
%
%     catch
%         burstToRemove(end+1) = i;
%     end
% end
% toc
%
% burstVals(burstToRemove)=[];
% xCenter(burstToRemove)=[];
% yCenter(burstToRemove)=[];
% numSpikesCon(burstToRemove)=[];
% peakSel(burstToRemove)=[];
% ampSel(burstToRemove)=[];
%
% % output params
% out.xCenter = xCenter;
% out.yCenter = yCenter;
% out.numSpikesCon = numSpikesCon;
% out.peakSel = peakSel;
% out.ampSel = ampSel;
% out.networkAct = networkAct;
% out.networkStats = networkStats;
