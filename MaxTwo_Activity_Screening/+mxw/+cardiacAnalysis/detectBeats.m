function out=detectBeats(tsSamples,deltaSamples,minNumSpikes)

% DETECTBEATS(tsSamples,deltaSamples,minNumSpikes)

%% group spikes based on temporal proximity

a = [inf; diff(tsSamples)];
% b = find(a<=deltaSamples);
d = find(a>deltaSamples);

gr=cell(0,0);

for i=1:(length(d)-1)
    
    iSt = d(i);
    iE = d(i+1)-1;
    
    if length(iSt:iE)>minNumSpikes
        gr{end+1}.iStart=iSt;
        gr{end}.iEnd=iE;
        gr{end}.ind=iSt:iE;
    end
end

out = gr;
