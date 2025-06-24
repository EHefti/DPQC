function [el, ind]=get_closest_electrode(POS, recording)

%[el, ind] = get_closest_electrode(POS, datainfo)
%  Returns closest available electrode (in datainfo) for a position POS

% x = recording.rawMap.map.x;
% y = recording.rawMap.map.y;
% els = recording.rawMap.map.electrode;

x = recording.rawMap.map.x;
y = recording.rawMap.map.y;
els = recording.rawMap.map.electrode;


d = mxw.util.pdist2(POS,[x y]);

[val ind]=min(d);

el = els(ind);
