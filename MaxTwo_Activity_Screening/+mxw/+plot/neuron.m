function neuron(neur, varargin)

%neuron Plots neurons features on a 2D-map
%
%  neuron(neur, 'doLines', true) plots STA waveforms (Default: true)
%
%  neuron(neur, 'doDots', true) plot dots at electrode positions
% 
% Input parameters:
%
%    -'WaveformColor': waveforms color
%    -'WaveformWidth': waveforms trace width



p = inputParser;

p.addParameter('doLines', true);
p.addParameter('doDots', true);
p.addParameter('WaveformSize', 14);
p.addParameter('WaveformColor', [0.1 0.1 0.1]);
p.addParameter('WaveformWidth', 1.2);

% p.addParameter('scatterSize', 100);

p.parse(varargin{:});

args = p.Results;




args.uVolts2uMetersScale = 10;


% args.PointSize = 100;


x = neur.x';
y = neur.y';


noBorder = 0;


hold on


if args.doLines
    
    waveforms = neur.template;
    
    normalizedWaveforms = waveforms ./ max(max(abs(waveforms)));
    
    %     if args.NormalizeByLocalMax
    %         normalizedWaveforms = waveforms ./ repmat(max(abs(waveforms)), size(waveforms, 1), 1);
    %     end
    %
    step = args.WaveformSize / size(normalizedWaveforms, 1);
    left = (x - args.WaveformSize/2)';
    
    waveforms2Plot = (-normalizedWaveforms * args.uVolts2uMetersScale) + repmat(y, size(normalizedWaveforms,1), 1);
    
    tempStep = cumsum(repmat(step, size(normalizedWaveforms, 1) + 1, size(normalizedWaveforms, 2)), 1);
    left2Rigth = repmat(left', size(normalizedWaveforms, 1), 1) + tempStep(1:end-1, :);
    
    plot(left2Rigth, waveforms2Plot, 'color', args.WaveformColor, 'Linewidth', args.WaveformWidth)
    
end


if args.doDots
    
    plot(neur.x,neur.y,'.','color',[0.4 0.4 0.4])
    
end

axis ij
axis equal

if ~noBorder
    border=11.3583;                         % adjust to like plot_neurons (?)
    xlim([min(x)-border max(x)+border])
    ylim([min(y)-border max(y)+border])
end

