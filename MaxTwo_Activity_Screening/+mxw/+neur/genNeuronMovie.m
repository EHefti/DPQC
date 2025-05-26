function genNeuronMovie(currNeur,mov,varargin)


%% parameters

doScalebar = 0;
doTxt = 1;


%%

close all;

% borders
xlm=min(currNeur.x);
xlma=max(currNeur.x);
ylm=min(currNeur.y);
ylma=max(currNeur.y);


firstSample = 1;
lastSample = size(currNeur.template,1);

%adjust minimum and maximum values of the colorbar (in uV)
minVal = min(min(currNeur.template));
maxVal = max(max(currNeur.template));

absVal = max([abs(minVal) abs(maxVal)]);

mini = absVal;
maxi = absVal;

PathName = cd;

% colormap
load('cmap_bluered.mat')


dateStr=char(datetime('now'));
dirName = [PathName '/' dateStr];
mkdir(dirName)


%% do images

cnt = 0;

for j=firstSample:lastSample
    
    j
    clims=[-mini,maxi];
    colormap(mycmap./256)
    
    plot_2D_map_clean(currNeur.x, currNeur.y, currNeur.template(j,:), clims, 'nearest');
    xlabel('\mum');ylabel('\mum');axis equal;
    
    colorbar
    
    %         set(gca,'XTickLabel', '')
    %         set(gca,'YTickLabel', '')
    hold all
    
    % scalebar
    if doScalebar
        xline = [max(currNeur.x)-200,max(currNeur.x)-100]; % 0.2 mm scale bar
        yline = [max(currNeur.y)-10,max(currNeur.y)-10];
        pl = line (xline,yline,'Color','w','LineWidth',5); % show scale bar
    end
    
    if doTxt
        txt = [sprintf('  %3.3f ms', (round((j-firstSample)/0.020)/1000))];
        %       txt = sprintf('%d ms', round((j-firstSample)/20));
        text(xlm+((xlma-xlm)/20),ylm+((ylma-ylm)/20),txt,'Color','w','FontSize',14); %show time
    end
    
    hold off
    xlim([xlm xlma])
    ylim([ylm ylma])
    
    pictureName = [sprintf('%03d',cnt)];
    
    savepng( 'Directory', dirName , 'FileName' , pictureName );
    
    cnt = cnt+1;
end

close(gcf)


%%

clear myVideo

myVideo = VideoWriter([mov '.avi'])
myVideo.FrameRate = 15;  % Default 30

open(myVideo);

for m= 0:(cnt-1)
    
    if m < 10
        mm1=strcat('00',int2str(m));
    elseif m < 100
        mm1=strcat('0',int2str(m));
    else
        mm1=int2str(m);
    end
    disp([mm1 '.png'])
    A = imread([dirName '/' mm1 '.png']);
    
    writeVideo(myVideo,A)
    
end

close(myVideo);

