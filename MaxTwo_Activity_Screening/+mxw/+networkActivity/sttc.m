%% STTC variable definition

function [ sttc_final ] = sttc( spiketrain1, spiketrain2, dt, TotRecTime )

spikeMat = [];

spikeMat(1,:) = histcounts(spiketrain1,0:1:TotRecTime);
spikeMat(2,:) = histcounts(spiketrain2,0:1:TotRecTime);


%Generate random spike trains
%[spikeMat, tVec] = poissonSpikeGen(100, 1, 2);
%figure; plotRaster(spikeMat, tVec*1000);
%xlabel('Time (ms)');
%ylabel('Trial Number');


%dt = 5; %ms +-dt: interval around spikes in which events are considered correlated
%TotRecTime = 1000; %ms     standard:1000
%STTC=0.5*[(Pa-Tb)/(1-Pa*Tb) + (Pb-Ta)/1-Pb*Ta)]

SpikeTrainA=[];
SpikeTrainB=[];
SpikeEvA=[];
SpikeEvB=[];

SpikeTrainA=spikeMat(1,:);
SpikeTrainB=spikeMat(2,:);

%find indices (times in ms) of spikes in trains
SpikeEvA=find(SpikeTrainA);
SpikeEvB=find(SpikeTrainB);



% Ta and Tb

%Ta: sum of all recording times within +-dt from any spike of A/Tot
%recording time
%Tb: sum of all recording times within +-dt from any spike of B/Tot
%recording time
%time intervals of spikes closer than |dt| are not considered twice

%find Ta
AdtLimits=[];
AdtLimitsNZ=[];
alreadyInA=[];
alreadyInA(1)=0;
Ta=[];
TaLim=[];
%check on events existance in SpikeTrain
if size(SpikeEvA,2)>0
    
    for i=1:(size(SpikeEvA,2)-1) %everything except last event
        
        %not overlapping intervals
        if ((SpikeEvA(i+1)-dt)-(SpikeEvA(i)+dt) > 0) && (SpikeEvA(i)>max(alreadyInA)) %second condition to avoid getting extreme of a cluster of spikes
            
            %events happening between 0 and dt
            if SpikeEvA(i)<=dt
                %left limits
                AdtLimits(1,i)=1;
                %right limits
                AdtLimits(2,i)=SpikeEvA(i)+dt;
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i);
                %events happening after (end-dt)
            elseif SpikeEvA(i)>=(size(spikeMat,2)-dt)
                %left limits
                AdtLimits(1,i)=SpikeEvA(i)-dt;
                %right limits
                AdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
                %events happening between (start + dt) and (end-dt)
            else
                %left limits
                AdtLimits(1,i)=SpikeEvA(i)-dt;
                %right limits
                AdtLimits(2,i)=SpikeEvA(i)+dt;
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
            end
            %overlapping intervals
        elseif (SpikeEvA(i+1)-dt)-(SpikeEvA(i)+dt) <= 0 && (SpikeEvA(i)>max(alreadyInA))
            %events happening before dt
            if SpikeEvA(i)<=dt
                %left limits
                AdtLimits(1,i)=1;
                %right limits
                for k=i:(size(SpikeEvA,2)-1) %k=i+1%%%%%%%%%%%%%%%%%%%%%%%%%
                    %keep iterating until events are spaced > 2*dt
                    if (SpikeEvA(k+1)-dt)-(SpikeEvA(k)+dt) <= 0 && k~=(size(SpikeEvA,2)-1)
                        
                        continue
                        %do nothing
                        %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        
                        %when (k+1)-th event is further that 2*dt from k-th event,
                        %stop iterating and store right limit
                    elseif  (SpikeEvA(k+1)-dt)-(SpikeEvA(k)+dt) > 0 && k~=(size(SpikeEvA,2)-1)  %%%
                        AdtLimits(2,i)=SpikeEvA(k)+dt;
                        %store last event included in the same interval to avoid
                        %counting it twice
                        alreadyInA(i)=SpikeEvA(k);
                        break
                    else
                        if SpikeEvA(k+1)>=(size(spikeMat,2)-dt)
                            AdtLimits(2,i)=size(spikeMat,2);
                            %store last event included in the same interval to avoid
                            %counting it twice
                            alreadyInA(i)=SpikeEvA(k+1);
                        else
                            AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                            alreadyInA(i)=SpikeEvA(k+1);
                            break
                        end
                        
                    end
                end
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i);
                
                %events happening after (end-dt)
            elseif SpikeEvA(i)>=(size(spikeMat,2)-dt)
                %left limits
                for k=i:-1:1
                    %keep iterating until events are spaced > 2*dt
                    if (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) <= 0 && ismember(SpikeEvA(k),alreadyInA)==0 && ismember(SpikeEvA(k-1),alreadyInA)==0
                        
                        continue
                        %do nothing
                        %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        
                        %when (k+1)-th event is further that 2*dt from k-th event,
                        %stop iterating and store left limit
                    elseif (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) > 0 %&& ismember(SpikeEvA(k),alreadyIn)==0 && ismember(SpikeEvA(k-1),alreadyIn)==0
                        AdtLimits(1,i)=SpikeEvA(k)-dt;
                        
                        %store last event included in the same interval to avoid
                        %counting it twice
                        % alreadyIn(i)=SpikeEvA(k);
                        break
                        %                 elseif (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) <= dt && ismember(SpikeEvA(k),alreadyIn)==1
                        %                    % AdtLimits(1,i)=AdtLimits(1,i-1);
                        %                     break
                        %                     %do nothing, it has already been stored
                    end
                end
                %right limits
                AdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
                
                %events happening between (start + dt) and (end-dt)
            else
                
                %value not already considered
                if ismember(SpikeEvA(i),alreadyInA)==0
                    %left limits
                    AdtLimits(1,i)=SpikeEvA(i)-dt;
                    
                    
                    %right limits
                    for k=i:(size(SpikeEvA,2)-1)
                        %keep iterating until events are spaced > 2*dt
                        if (SpikeEvA(k+1)-dt)-(SpikeEvA(k)+dt) <= 0 && k~=(size(SpikeEvA,2)-1)
                            
                            continue
                            %do nothing
                            %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        elseif (SpikeEvA(k+1)-dt)-(SpikeEvA(k)+dt) <= 0 && k==(size(SpikeEvA,2)-1)
                            if size(spikeMat,2)-SpikeEvA(k+1)> dt
                                AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                                alreadyInA(i)=SpikeEvA(k+1);
                            else
                                AdtLimits(2,i)=size(spikeMat,2);
                                alreadyInA(i)=SpikeEvA(k+1);
                            end
                            %when (k+1)-th event is further that 2*dt from k-th event,
                            %stop iterating and store right limit
                        else
                            AdtLimits(2,i)=SpikeEvA(k)+dt;
                            %store last event included in the same interval to avoid
                            %counting it twice
                            alreadyInA(i)=SpikeEvA(k);
                            break
                        end
                    end
                    
                    
                    %section within dt around spike
                    TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
                    
                end
            end
            
        elseif SpikeEvA(i)<max(alreadyInA)
            continue
        end
    end
    
    for i=size(SpikeEvA,2) %last event
        
        %not overlapping intervals
        if (SpikeEvA(i)-dt)-(SpikeEvA(i-1)+dt) > 0
            
            %events happening between 0 and dt
            if SpikeEvA(i)<=dt
                %left limits
                AdtLimits(1,i)=1;
                %right limits
                AdtLimits(2,i)=SpikeEvA(i)+dt;
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i);
                
                %events happening after (end-dt)
            elseif SpikeEvA(i)>=(size(spikeMat,2)-dt)
                %left limits
                AdtLimits(1,i)=SpikeEvA(i)-dt;
                %right limits
                AdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
                
                
                %events happening between (start + dt) and (end-dt)
            else
                %left limits
                AdtLimits(1,i)=SpikeEvA(i)-dt;
                %right limits
                AdtLimits(2,i)=SpikeEvA(i)+dt;
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
            end
            
            %%%%%%
            %overlapping intervals
        elseif (SpikeEvA(i)-dt)-(SpikeEvA(i-1)+dt) <= 0 && (SpikeEvA(i)>max(alreadyInA))
            %events happening before dt
            if SpikeEvA(i)<=dt
                %left limits
                AdtLimits(1,i)=1;
                %right limits
                AdtLimits(2,i)=SpikeEvA(i)+dt;
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i);
                
                %events happening after (end-dt)
            elseif SpikeEvA(i)>=(size(spikeMat,2)-dt)
                %left limits
                for k=i:-1:1
                    %keep iterating until events are spaced > 2*dt
                    if (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) <= 0 && ismember(SpikeEvA(k),alreadyInA)==0 && ismember(SpikeEvA(k-1),alreadyInA)==0
                        
                        continue
                        %do nothing
                        %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        
                        %when (k+1)-th event is further that 2*dt from k-th event,
                        %stop iterating and store left limit
                    elseif (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) > 0 %&& ismember(SpikeEvA(k),alreadyIn)==0 && ismember(SpikeEvA(k-1),alreadyIn)==0
                        AdtLimits(1,i)=SpikeEvA(k)-dt;
                        
                        %store last event included in the same interval to avoid
                        %counting it twice
                        % alreadyIn(i)=SpikeEvA(k);
                        break
                        %                 elseif (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) <= dt && ismember(SpikeEvA(k),alreadyIn)==1
                        %                    % AdtLimits(1,i)=AdtLimits(1,i-1);
                        %                     break
                        %                     %do nothing, it has already been stored
                    end
                end
                %right limits
                AdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
                
                %events happening between (start + dt) and (end-dt)
            else
                
                %value not already considered
                if ismember(SpikeEvA(i),alreadyInA)==0
                    %left limits
                    AdtLimits(1,i)=SpikeEvA(i)-dt;
                    
                    
                    %right limits
                    AdtLimits(2,i)=SpikeEvA(i)+dt;
                    
                    
                    %section within dt around spike
                    TaLim(i)=AdtLimits(2,i)- AdtLimits(1,i);
                    
                end
            end
            
        elseif SpikeEvA(i)<max(alreadyInA)
            continue
        end
        %%%%%%
        
        
        
        
    end
    
    
    %limits of the tiles around events
    AdtLimitsNZ(1,:)=nonzeros(AdtLimits(1,:));
    AdtLimitsNZ(2,:)=nonzeros(AdtLimits(2,:));
    
    %Ta
    Ta=sum(TaLim)/TotRecTime; %ms
end
%find Tb
BdtLimits=[];
BdtLimitsNZ=[];
alreadyInB=[];
alreadyInB(1)=0;
Tb=[];
TbLim=[];
%check on events existance in SpikeTrain
if size(SpikeEvB,2)>0
    for i=1:(size(SpikeEvB,2)-1) %everything except last event
        
        %not overlapping intervals
        if ((SpikeEvB(i+1)-dt)-(SpikeEvB(i)+dt) > 0) && (SpikeEvB(i)>max(alreadyInB)) %second condition to avoid getting extreme of a cluster of spikes
            
            %events happening between 0 and dt
            if SpikeEvB(i)<=dt
                %left limits
                BdtLimits(1,i)=1;
                %right limits
                BdtLimits(2,i)=SpikeEvB(i)+dt;
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i);
                %events happening after (end-dt)
            elseif SpikeEvB(i)>=(size(spikeMat,2)-dt)
                %left limits
                BdtLimits(1,i)=SpikeEvB(i)-dt;
                %right limits
                BdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
                %events happening between (start + dt) and (end-dt)
            else
                %left limits
                BdtLimits(1,i)=SpikeEvB(i)-dt;
                %right limits
                BdtLimits(2,i)=SpikeEvB(i)+dt;
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
            end
            %overlapping intervals
        elseif (SpikeEvB(i+1)-dt)-(SpikeEvB(i)+dt) <= 0 && (SpikeEvB(i)>max(alreadyInB))
            %events happening before dt
            if SpikeEvB(i)<=dt
                %left limits
                BdtLimits(1,i)=1;
                %right limits
                for k=i:(size(SpikeEvB,2)-1)   %k=i+1%%%%%%%%%%%%%%%%%%%%%%%%%
                    %keep iterating until events are spaced > 2*dt
                    if (SpikeEvB(k+1)-dt)-(SpikeEvB(k)+dt) <= 0 && k~=(size(SpikeEvB,2)-1)   % <= 0 !!!!
                        
                        continue
                        %do nothing
                        %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        
                        %when (k+1)-th event is further that 2*dt from k-th event,
                        %stop iterating and store right limit
                    elseif  (SpikeEvB(k+1)-dt)-(SpikeEvB(k)+dt) > 0 && k~=(size(SpikeEvB,2)-1)
                        BdtLimits(2,i)=SpikeEvB(k)+dt;
                        %store last event included in the same interval to avoid
                        %counting it twice
                        alreadyInB(i)=SpikeEvB(k);
                        break
                    else
                        if SpikeEvB(k+1)>=(size(spikeMat,2)-dt)
                            
                            BdtLimits(2,i)=size(spikeMat,2);
                            %store last event included in the same interval to avoid
                            %counting it twice
                            alreadyInB(i)=SpikeEvB(k+1);
                        else
                            
                            BdtLimits(2,i)=SpikeEvB(k+1)+dt;
                            alreadyInB(i)=SpikeEvB(k+1);
                            break
                        end
                    end
                end
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i);
                
                %events happening after (end-dt)
            elseif SpikeEvB(i)>=(size(spikeMat,2)-dt)
                %left limits
                for k=i:-1:1
                    %keep iterating until events are spaced > 2*dt
                    if (SpikeEvB(k)-dt)-(SpikeEvB(k-1)+dt) <= 0 && ismember(SpikeEvB(k),alreadyInB)==0 && ismember(SpikeEvB(k-1),alreadyInB)==0
                        
                        continue
                        %do nothing
                        %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        
                        %when (k+1)-th event is further that 2*dt from k-th event,
                        %stop iterating and store left limit
                    elseif (SpikeEvB(k)-dt)-(SpikeEvB(k-1)+dt) > 0 %&& ismember(SpikeEvA(k),alreadyIn)==0 && ismember(SpikeEvA(k-1),alreadyIn)==0
                        BdtLimits(1,i)=SpikeEvB(k)-dt;
                        
                        %store last event included in the same interval to avoid
                        %counting it twice
                        % alreadyIn(i)=SpikeEvA(k);
                        break
                        %                 elseif (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) <= dt && ismember(SpikeEvA(k),alreadyIn)==1
                        %                    % AdtLimits(1,i)=AdtLimits(1,i-1);
                        %                     break
                        %                     %do nothing, it has already been stored
                    end
                end
                %right limits
                BdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
                
                %events happening between (start + dt) and (end-dt)
            else
                
                %value not already considered
                if ismember(SpikeEvB(i),alreadyInB)==0
                    %left limits
                    BdtLimits(1,i)=SpikeEvB(i)-dt;
                    
                    
                    %right limits
                    for k=i:(size(SpikeEvB,2)-1)  %k=i+1%%%%%%%%%%%%%%%%%%%%%%%%%
                        %keep iterating until events are spaced > 2*dt
                        if (SpikeEvB(k+1)-dt)-(SpikeEvB(k)+dt) <= 0 && k~=(size(SpikeEvB,2)-1)
                            
                            continue
                            %do nothing
                            %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                            
                        elseif (SpikeEvB(k+1)-dt)-(SpikeEvB(k)+dt) <= 0 && k==(size(SpikeEvB,2)-1)
                            if size(spikeMat,2)-SpikeEvB(k+1)> dt
                                BdtLimits(2,i)=SpikeEvB(k+1)+dt;
                                alreadyInB(i)=SpikeEvB(k+1);
                            else
                                BdtLimits(2,i)=size(spikeMat,2);
                                alreadyInB(i)=SpikeEvB(k+1);
                            end
                            %when (k+1)-th event is further that 2*dt from k-th event,
                            %stop iterating and store right limit
                        else
                            BdtLimits(2,i)=SpikeEvB(k)+dt;
                            %store last event included in the same interval to avoid
                            %counting it twice
                            alreadyInB(i)=SpikeEvB(k);
                            break
                        end
                    end
                    
                    
                    %section within dt around spike
                    TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
                    
                end
            end
            
        elseif SpikeEvB(i)<max(alreadyInB)
            continue
        end
    end
    
    for i=size(SpikeEvB,2) %last event
        
        %not overlapping intervals
        if (SpikeEvB(i)-dt)-(SpikeEvB(i-1)+dt) > 0
            
            %events happening between 0 and dt
            if SpikeEvB(i)<=dt
                %left limits
                BdtLimits(1,i)=1;
                %right limits
                BdtLimits(2,i)=SpikeEvB(i)+dt;
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i);
                
                %events happening after (end-dt)
            elseif SpikeEvB(i)>=(size(spikeMat,2)-dt)
                %left limits
                BdtLimits(1,i)=SpikeEvB(i)-dt;
                %right limits
                BdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
                
                
                %events happening between (start + dt) and (end-dt)
            else
                %left limits
                BdtLimits(1,i)=SpikeEvB(i)-dt;
                %right limits
                BdtLimits(2,i)=SpikeEvB(i)+dt;
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
            end
            
            %%%%%%
            %overlapping intervals
        elseif (SpikeEvB(i)-dt)-(SpikeEvB(i-1)+dt) <= 0 && (SpikeEvB(i)>max(alreadyInB))
            %events happening before dt
            if SpikeEvB(i)<=dt
                %left limits
                BdtLimits(1,i)=1;
                %right limits
                BdtLimits(2,i)=SpikeEvB(i)+dt;
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i);
                
                %events happening after (end-dt)
            elseif SpikeEvB(i)>=(size(spikeMat,2)-dt)
                %left limits
                for k=i:-1:1
                    %keep iterating until events are spaced > 2*dt
                    if (SpikeEvB(k)-dt)-(SpikeEvB(k-1)+dt) <= 0 && ismember(SpikeEvB(k),alreadyInB)==0 && ismember(SpikeEvB(k-1),alreadyInB)==0
                        
                        continue
                        %do nothing
                        %AdtLimits(2,i)=SpikeEvA(k+1)+dt;
                        
                        %when (k+1)-th event is further that 2*dt from k-th event,
                        %stop iterating and store left limit
                    elseif (SpikeEvB(k)-dt)-(SpikeEvB(k-1)+dt) > 0 %&& ismember(SpikeEvA(k),alreadyIn)==0 && ismember(SpikeEvA(k-1),alreadyIn)==0
                        BdtLimits(1,i)=SpikeEvB(k)-dt;
                        
                        %store last event included in the same interval to avoid
                        %counting it twice
                        % alreadyIn(i)=SpikeEvA(k);
                        break
                        %                 elseif (SpikeEvA(k)-dt)-(SpikeEvA(k-1)+dt) <= dt && ismember(SpikeEvA(k),alreadyIn)==1
                        %                    % AdtLimits(1,i)=AdtLimits(1,i-1);
                        %                     break
                        %                     %do nothing, it has already been stored
                    end
                end
                %right limits
                BdtLimits(2,i)=size(spikeMat,2);
                %section within dt around spike
                TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
                
                %events happening between (start + dt) and (end-dt)
            else
                
                %value not already considered
                if ismember(SpikeEvB(i),alreadyInB)==0
                    %left limits
                    BdtLimits(1,i)=SpikeEvB(i)-dt;
                    
                    
                    %right limits
                    BdtLimits(2,i)=SpikeEvB(i)+dt;
                    
                    
                    %section within dt around spike
                    TbLim(i)=BdtLimits(2,i)- BdtLimits(1,i);
                    
                end
            end
            
        elseif SpikeEvB(i)<max(alreadyInB)
            continue
        end
        %%%%%%
        
        
        
        
    end
    
    
    
    %limits of the tiles around events
    BdtLimitsNZ(1,:)=nonzeros(BdtLimits(1,:));
    BdtLimitsNZ(2,:)=nonzeros(BdtLimits(2,:));
    
    %Tb
    Tb=sum(TbLim)/TotRecTime; %ms
end

% Pa and Pb

%Pa: (spikes of A that lie within +- dt of any spike from B)/(tot. n. of
%spikes of A)
%Pb: (spikes of B that lie within +- dt of any spike from A)/(tot. n. of
%spikes of B)

%find Pa
Pa=[];
SpikePa=[];

if size(SpikeEvA,2)>0
    for i=1:length(SpikeEvA)
        for j=1:size(BdtLimitsNZ,2)
            if (BdtLimitsNZ(1,j) <= SpikeEvA(i)) && (SpikeEvA(i) <= BdtLimitsNZ(2,j))
                
                SpikePa=[SpikePa SpikeEvA(i)];
            else
                continue
            end
        end
        
    end
    
    Pa=size(SpikePa,2)/size(SpikeEvA,2);
end
%find Pb
Pb=[];
SpikePb=[];
if size(SpikeEvB,2)>0
    for i=1:length(SpikeEvB)
        for j=1:size(AdtLimitsNZ,2)
            if (AdtLimitsNZ(1,j) <= SpikeEvB(i)) && (SpikeEvB(i) <= AdtLimitsNZ(2,j))
                
                SpikePb=[SpikePb SpikeEvB(i)];
            else
                continue
            end
        end
        
    end
    
    Pb=size(SpikePb,2)/size(SpikeEvB,2);
end
% STTC calculation

STTC = [];

if size(SpikeEvA,2)>0 && size(SpikeEvB,2)>0
    % STTC has values between 1 (max correlation) and -1 (no correlation)
    STTC = 0.5*(((Pa-Tb)/(1-Pa*Tb)) + ((Pb-Ta)/(1-Pb*Ta)));
else
    sprintf('Impossible to compare the two spike trains. Check inputs.')
end

sttc_final = STTC;

end



