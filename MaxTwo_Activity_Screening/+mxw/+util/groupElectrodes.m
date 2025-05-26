function group=groupElectrodes(POS,dist)
% 
%GROUP_ELECTRODES group electrodes according to distance
%
%   G=GROUP_ELECTRODES(POS, dist) POS: electrode positions. The output G
%   is a cell and has the same size as there were found groups of electrodes.
%   The cells contain the indices of electrodes belonging to that group.
%   Set the distance threshold for grouping.
%   Electrodes will be taken into one group, if their distance is smaller than
%   dist. 



% check size
if size(POS,1)<=1 || size(POS,2)~=2
    error('\ncheck the input!');
end



% compute all pairwise distances

Z = mxw.util.pdist2(POS,POS,'euclidean');

Z(Z==0)=inf;
[I,J]=find(Z<dist);
pairs=[I J];
orig_pairs=pairs;

% electrodes that are NOT in pairs:
not_in_pairs=setdiff(size(POS,1),unique(orig_pairs));
group=cell(0,0);

for i=1:length(not_in_pairs)
    group{i}=not_in_pairs(i);
end

if ~isempty(pairs)
    
    i=length(group)+1;    % group index
    eog=0;
    
    while ~eog
        
        % first action: assign first pair to new group
        group{i}=pairs(1,:);
        pairs(1,:)=[];
        
        j=1; %
        while j<=length(group{i})
            
            [r,c]=find(pairs==group{i}(j));
            
            if isempty(r)
                % do nothing
                j=j+1;
            else
                to_group=pairs(r,:);
                
                group{i}=[group{i} to_group(:)'];
                pairs(r,:)=[];
                group{i}=unique(group{i});
                j=1;
            end
            
        end
        % finnished group i
        i=i+1;
        if isempty(pairs)
            eog=1;
        end
    end
end
