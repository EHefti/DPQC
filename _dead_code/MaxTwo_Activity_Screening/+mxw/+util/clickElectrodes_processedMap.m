function el = clickElectrodes_processedMap(datainfo)

%clickElectrodes(datainfo) 
%  Click on positions on array plot -> closest electrodes get labeled
%
%  el = clickElectrodes(datainfo) returnds a list with the selected
%  electrodes

% map = datainfo.rawMap.map;

map = datainfo.processedMap;

[x y] = ginput;

for i=1:length(x)
    
    el(i)=double(mxw.util.get_closest_electrode_processedMap([x(i) y(i)],datainfo));
    
    el_ind=find(map.electrode==el(i));
    
    
    hold on
    text(map.xpos(el_ind),map.ypos(el_ind)-1,num2str(el(i)), 'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Bottom', 'FontSize', 9, 'Color', [0.2 0.2 0.2]);
    
end
    
el = unique(el,'stable');
