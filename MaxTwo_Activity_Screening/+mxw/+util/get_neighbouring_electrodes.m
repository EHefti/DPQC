function [neighEls, neighInd]=get_neighbouring_electrodes(fixEl, recording, nEls)

% 


x = recording.rawMap.map.x;
y = recording.rawMap.map.y;
els = recording.rawMap.map.electrode;

fixElInd = find(recording.rawMap.map.electrode==fixEl);

POS = [x(fixElInd) y(fixElInd)];

d = mxw.util.calculateEuclideanDist(POS,[x y]);

[val, ind]=sort(d);

if val(1)>0
    error('FixEl not in recording file')
end

neighInd = ind(2:(nEls+1));
neighEls = els(ind(2:(nEls+1)));
