function [ populatedSelection ] = populateSelec( selection, mposx, mposy, minNeighbourElectrodes )

populatedSelection = zeros(length(mposx),1);
populatedSelection(selection) = 1;
selectedConfig = selection;

for i = 1:length(selection)
    neighbours = (mposx == mposx(selectedConfig(i)) | mposx == mposx(selectedConfig(i))+1 | mposx == mposx(selectedConfig(i))-1) & ...
        (mposy == mposy(selectedConfig(i)) | mposy == mposy(selectedConfig(i))+1 | mposy == mposy(selectedConfig(i))-1);
    totalNeigh = sum(neighbours & populatedSelection);
    
    if totalNeigh < minNeighbourElectrodes
        neighbours = (mposx == mposx(selectedConfig(i)) ...
            & (mposy == (mposy(selectedConfig(i))-1) | mposy == mposy(selectedConfig(i))+1))...
            +(mposy == mposy(selectedConfig(i)) ...
            & (mposx == (mposx(selectedConfig(i))-1) | mposx == mposx(selectedConfig(i))+1));
        
        populatedSelection = populatedSelection + neighbours;
    end
end
end
