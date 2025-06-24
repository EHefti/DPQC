function selectionDistributions( spikeCount, amplitude, varCoeff, selecIdx )
    % SELECTIONDISTRIBUTIONS plots the electrodes used and selected in 
    % 'mxw.util.electrodeSelection...'. This plotting function should not
    % be used directly by the user since it is internally called in 
    % 'mxw.util.electrodeSelection...'.
    %
    %

figure;
subplot(1,3,1);
hold on;
set(gca,'FontSize',14);
y1 = spikeCount;
x1 = (ones(1,length(y1))) + (-0.3 + (0.3 + 0.3)*rand(length(y1),1))';
plot(x1, y1, 'o', 'color', [0, 0.7, 0]);
plot(x1(selecIdx), y1(selecIdx), 'x', 'color', 'r');
title('Activity distribution', 'FontName', 'Source Sans Pro');
ylabel('Firing rate [#peaks m^{-1}] ', 'FontName', 'Source Sans Pro')
legend('Whole Array', 'Selection')

subplot(1,3,2);
hold on;
set(gca,'FontSize',14);
y2 = -amplitude;
x2 = (ones(1,length(y2))) + (-0.3 + (0.3 + 0.3)*rand(length(y2),1))';
plot(x2, y2, 'o', 'color', [0, 0.7, 0]);
plot(x2(selecIdx), y2(selecIdx), 'x', 'color', 'r');
title('Amplitude distribution', 'FontName', 'Source Sans Pro')
ylabel('Amlitude (#uV)', 'FontName', 'Source Sans Pro')
legend('Whole Array', 'Selection')

subplot(1,3,3);
hold on;
set(gca,'FontSize',14);
y3 = -varCoeff;
x3 = (ones(1,length(y3))) + (-0.3 + (0.3 + 0.3)*rand(length(y3),1))';
plot(x3, y3, 'o', 'color', [0, 0.7, 0]);
plot(x3(selecIdx), y3(selecIdx), 'x', 'color', 'r');
title('Variational coefficient of the amplitude', 'FontName', 'Source Sans Pro')
ylabel('Peak std/Mean peak amplitude', 'FontName', 'Source Sans Pro')
legend('Whole Array', 'Selection')
end