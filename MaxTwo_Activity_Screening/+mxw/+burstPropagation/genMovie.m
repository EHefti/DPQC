function genMovie(burstVals,burstNum,binSize,dirName,recName)

% generate Burst Propagation Movie


% binSize = 5; % [ms]


% BURST PROPAGATION MOVIE

mkdir(dirName)

%%
% input: tsPerChannel

spikes = burstVals{burstNum}.tsPerCh;

xAll=burstVals{burstNum}.xAll;
yAll=burstVals{burstNum}.yAll;


chCol = spikes(:,1);
tsCol = spikes(:,2)*1000; % [ms]
tsCol2 = tsCol - min(tsCol);
xCol =  spikes(:,3);
yCol =  spikes(:,4);

% histogram edges:
t = 0:binSize:max(tsCol2)

[C,ia,ic] = unique(chCol);

binendTs = zeros(length(t)-1,length(ia));
binendTsHz = zeros(length(t)-1,length(ia));

c=1;
tic
for i = 1:length(ia)
    
    thisChInd = find(chCol==chCol(ia(i)));
    h=histogram(tsCol2(thisChInd),t);
    binendTs(:,c)=h.Values;
    binendTsHz(:,c)=h.Values/binSize*1000;
    x(c)=xCol(ia(i));
    y(c)=yCol(ia(i));
    c=c+1;
end
close(gcf)

%% Plot movie images

scalebarLen = 500; % [um]
scalebarD = 60; % distance from border
scalebarPos = 'bottomLeft'
textDx = 400;
textDy = 100;
markerSize = 20;

maxC = max(binendTsHz(:));

xmin=min(xAll);
xmax=max(xAll);

ymin=min(yAll);
ymax=max(yAll);

%border
b=50

close all

% load('cmap_bluered.mat')

xx = 0;
figure('color','w','Position',[680 639 980 459]);

for j=[1 1 1 1:size(binendTsHz,1)]
    
    
    
    xx=xx+1;
    
    
%     colormap(mycmap./256)
    %         plot_2D_map_clean(x, y, binendTsHz(j,:), clims);
    
    
    scatter(xAll,yAll,markerSize,[0.7 0.7 0.7],'s');
    
        scatter(x,y,markerSize,binendTsHz(j,:),'s','filled');
    hold on
    col=colormap('hot');
    col = flipud(col);
    colormap(col);
        %         xlabel('\mum');ylabel('\mum');
    axis equal;
    axis ij
    
    c=colorbar;
%     set(gca,'XTickLabel', '')
%     set(gca,'YTickLabel', '')
    
    hAx = gca;
    hAx.CLim = [0 maxC]; %
      hold on
   scatter(xAll,yAll,markerSize,[0.9 0.9 0.9],'s');
   
    
    %         plot(map2.x, map2.y,'w.')
    %
    if strcmp(scalebarPos,'topRight')
        xline = [xmax-scalebarD-scalebarLen,xmax-scalebarD]; % 0.2 mm scale bar
        yline = [ymin+scalebarD,ymin+scalebarD];
        
    elseif strcmp(scalebarPos,'bottomLeft')
        
        xline = [xmin+scalebarD,xmin+scalebarD+scalebarLen]; % 0.2 mm scale bar
        yline = [ymax-scalebarD,ymax-scalebarD];
    end
    
    pl = line (xline,yline,'Color','k','LineWidth',5); % show scale bar
    
    text(mean(pl.XData),mean(pl.YData)-80,[int2str(scalebarLen) '\mum'],'Color','k','FontSize',14,'HorizontalAlignment','center'); %show time
    
    
    txt = sprintf('%3.0f ms', j*binSize);
    text(xmax-textDx,ymin+textDy,txt,'Color','k','FontSize',14); %show time
    %         hold off
    xlim([xmin-b xmax+b])
    ylim([ymin-b ymax+b])
    box on
    
	hold off
    
    mov(j) = getframe(gcf);
    
    
end



%%

clear myVideo

myVideo = VideoWriter([dirName '/' recName]);
myVideo.FrameRate = 4;  % Default 30

open(myVideo);

for m= 1:size(binendTsHz,1)
    

    if size(mov(m).cdata) == size(mov(end).cdata)
        try
            writeVideo(myVideo,mov(m));
        catch
            warning('Missing frames');
        end
        
    else
        
    end
end
close(myVideo);


