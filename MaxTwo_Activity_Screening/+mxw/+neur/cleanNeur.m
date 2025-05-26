function out = cleanNeur(neur)



%% Inputs: neuron, id, number of electrodes
%
% neur_id=13;

n_el=3;
split_thresh = 3; % split if difference in uV larger
do_plot=0;

%
% neur = neurs(neur_id)

smpls_template = size(neur.template,1);


[v, ind] = sort(max(neur.template)-min(neur.template),'desc');

%%

tr_per_ts=zeros(n_el*smpls_template,length(neur.ts));

for i=1:n_el
    
    el_ind = ind(i);
    
    tr=neur.traces{el_ind};
    
    ind_mat=(1:smpls_template)+(i-1)*smpls_template;
    
    tr_per_ts(ind_mat,:)=tr;
    
    
end


%% Feature AND Clustering

feature = 'PCA';
cluster_method = 'kmeans';

num_of_pca = 4
n_clust = 2



%% define feature

if strcmp(feature,'PCA')
    
    % feature
    [COEFF SCORE] = pca(tr_per_ts');
    data=SCORE(:,1:min([size(SCORE,2) num_of_pca]));
    
elseif strcmp(feature,'amplitude')
    
    % use pktopk amplitudes as features
    %     if ~isfield(n1,'peaks')
    n1b=get_pktopks({n1});
    n1=n1b{1};
    %     end
    pks=n1.peaks.trace_peaks;
    data=zeros(size(pks,1),max(1,size(pks,2)));
    data(:,1:size(pks,2))=pks;
elseif strcmp(feature,'amplitude_pca')
    
    % use pktopk amplitudes as features
    if ~isfield(n1,'peaks')
        n1b=get_pktopks({n1});
        n1=n1b{1};
    end
    pks=n1.peaks.trace_peaks;
    
    [COEFF, SCORE] = princomp(pks,'econ');
    data=SCORE(:,1:min([size(SCORE,2) num_of_pca]));
    
elseif strcmp(feature,'data')
    
    data=n1.tr_per_ts;
    
end


%% here cluster


switch cluster_method
    case 'KK'
        disp('Method is Klustakwik')
        if doOutlierFix
            T=runKlustaKwik(data,'klustakwik_tmp','11111111111111111111111111111111111111100','do_outlier_fix');
        else
            T=runKlustaKwik(data,'klustakwik_tmp','11111111111111111111111111111111111111100');
        end
    case 'kmeans'
        disp('Method is kmeans')
        T = kmeans(data,n_clust);
    case 'clusterdata'
        disp('Method is nearest')
        T = clusterdata(data,'distance','seuclidean','maxclust',n_clust);
    case 'gmm'
        gm = gmdistribution.fit(data,n_clust)
        T = cluster(gm,data);
    case 'fuzzy'
        [center, U, obj_fcn] = fcm(data, n_clust, [2 1000 1e-100 0]);
        [val ind] = max(U);
        T=ind;
    otherwise
        disp('No cluster Method defined!')
        return
end


%% visualize

unT = unique(T)

lin=lines;

if do_plot
    figure
    subplot(121)
    plot(SCORE(:,1),SCORE(:,2),'.')
    
    subplot(122)
    hold on
    
    for i=1:length(unT)
        
        a=find(T==unT(i))
        plot(tr_per_ts(:,a),'color',lin(i,:));
        
    end
end

%% Find larger cluster

clear b
for i=1:length(unT)
    
    a=find(T==unT(i))
    if do_plot
        plot(tr_per_ts(:,a),'color',lin(i,:));
    end
    b(:,i)=mean(tr_per_ts(:,a),2);
end

% select larger neuron, if diff in neg. pat larger 4 uV
if abs(diff(min(b)))>split_thresh
    
    [v ii]=min(min(b));
    clusterToKeep = ii;
    tsToKeep = find(T==ii);
    
    neur.ts = neur.ts(tsToKeep);
    neur.frame_no = neur.frame_no(tsToKeep);
    
    for i=1:length(neur.traces)
        neur.traces{i}=neur.traces{i}(:,tsToKeep);
        neur.template(:,i) = mean(neur.traces{1},2);
    end
    
end

out = neur;

