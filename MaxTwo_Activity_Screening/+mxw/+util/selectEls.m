function selected = selectEls(datainfo, N)

%% get electrodes with largest spikes

amp= zeros(size(datainfo.extractedSpikes.amplitude));

for i=1:length(datainfo.extractedSpikes.amplitude)
    
    if isempty(datainfo.extractedSpikes.amplitude{i})
        amp(i) = 0;
    else
        amp(i) = max(abs(datainfo.extractedSpikes.amplitude{i}));
    end
    
end

[~, in] = sort(amp,'descend');

els = double(datainfo.rawMap.map.electrode(in));
x = datainfo.rawMap.map.x(in);
y = datainfo.rawMap.map.y(in);

ind_remove = [];
for i = 1:length(els)
    
    POS = [x(i) y(i)];
    d = pdist2(POS,[x(1:(i-1)) y(1:(i-1))]);
    if min(d)<25
        
        x(i) = 1000000;
        y(i) = 1000000;
    end
end
del_ind = x==1000000;

els(del_ind)=[];
x(del_ind)=[]; 
y(del_ind)=[]; 


selected = els(1:min(N,length(els)));

