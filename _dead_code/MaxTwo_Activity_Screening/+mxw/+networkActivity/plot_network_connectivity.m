function plot_network_v010718MS(G,XY)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% G: Thresholded connectivity matrix (weighted or binary)
% XY: Coordinates for each electrode/node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some parameters to tweak the visualization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%figure('Units','pixels','Position',[200 200 600 300],'color','w');

plot_edges = true; % if you want to plot edges in general
plot_top_edges = false; % if you want to plot the top 5% edges in a different color (this works only, if you provide a weighted input matrix)

factor = 7; % scale up the size of each node x factor 
node_color = [254 178 76]./255; % orange
node_line_color = [0.1 0.1 0.1]; % dark grey
edge_color = [0.7 0.7 0.7]; % light grey
top_edge_color = [1 0 0]; % red

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the edges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(G); % number of nodes
edges = find(G>0); % indices of all edges
El = [];

for e = 1:length(edges)
  [i,j] = ind2sub([n,n],edges(e)); % node indices of edge e  
  El = [El; i j G(i,j)];
end

for i = 1:length(El)
    el(i,:)=[El(i,1) El(i,2) El(i,3) XY(El(i,1),1) XY(El(i,1),2) XY(El(i,2),1) XY(El(i,2),2)];
end

% Loop through edge list
for i = 1:size(el,1)
    
    % Plot line between two nodes
    x1=el(i,4); y1=el(i,5);
    x2=el(i,6); y2=el(i,7);
    
    % plot all edges (no weights)
    if plot_edges
        if el(i,3) > 0
            hold on; line([x1 x2],[y1 y2],'LineWidth',0.5,'Color',edge_color);
        end
        
        if plot_top_edges
            top_95 = mxw.util.percentile(nonzeros(el(:,3)),95);
            if el(i,3) >= top_95
                hold on; line([x1 x2],[y1 y2],'LineWidth',1.3,'Color',top_edge_color);
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the nodes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

G = abs(G); % check if you want this!
deg = sum(G>0)*factor; % degree of each node
active = find(sum(G)); % plot only nodes with degree > 0
hold on; scatter(XY(active,1),XY(active,2), deg(active), node_color, 'filled'); 
hold on; scatter(XY(active,1),XY(active,2), deg(active), node_line_color,'LineWidth',1.0); axis equal; axis ij; 
end