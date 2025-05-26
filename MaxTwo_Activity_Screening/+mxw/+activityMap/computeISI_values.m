function [ all_isi mean_isi_per_el ] = computeISI_values( fileManagerObj )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

nFiles = fileManagerObj.nFiles;
nElectrodes = length(fileManagerObj.processedMap.electrode);
tempValues = zeros(nElectrodes,1);
values = zeros(nElectrodes,1);

max_isi=200; %ms

all_isi = [];

for i = 1:nElectrodes
    isi_per_el{i}=[];
end
    
for iFile = 1:nFiles
    for i = 1:nElectrodes
        if ~isempty(fileManagerObj.extractedSpikes(iFile).frameno{i})
            spikes=double(fileManagerObj.extractedSpikes(iFile).frameno{i});
            if length(spikes)>1
                isi= diff(spikes)/20;
                isi_thr=isi(isi<max_isi);
                isi_per_el{i}=[isi_per_el{i} isi_thr];
                all_isi = [all_isi isi_thr];
            end
        end

    end
    
end

for i = 1:nElectrodes
    mean_isi_per_el(i)=mean(isi_per_el{i});
end

%%


