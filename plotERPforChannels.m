
addpath bva-io-master/
addpath (genpath('eeglab2022.1/'));
eeglab
load("dataset_ds005284.mat");

folder = 'EvokedPotentialPlots';
htmlGray = [128 128 128]/255;
channelLabels = {data{1}.chanlocs.labels};
for figi=1:length(channelLabels)
    gcf= figure();
    hold on
    averagedERP = zeros(size(data,2),1200);
    for i=1:size(data,2)
        disp (num2str(i))
        EEG = data{i};

        EEG_ep = pop_epoch(EEG, 'condition 54', [-0.20 1.0]);
        % baseline-correct
        EEG_ep = pop_rmbase(EEG_ep, [-200 0]);
    
        EEG_ep = pop_autorej(EEG_ep, 'threshold', 100, 'startprob', 5, 'maxrej', 5, 'nogui', 'on');
    
        
        ERP = mean(EEG_ep.data, 3);
        t = EEG_ep.times;
        averagedERP(i,:) = ERP(figi, :);
        plot(t, ERP(figi, :), 'LineWidth', 1, 'Color', htmlGray)

    end
    aveERP = mean(averagedERP, 1);
    plot(t, aveERP, 'LineWidth', 3,'Color', [0 0 0])
    xlabel('Time (ms)')
    ylabel('Amplitude (µV)')
    title(channelLabels{figi})
    hold off
    description = strcat("ERP_channel_", channelLabels{figi});
    set(gca,'FontSize',18)
    grid on
    filename = strcat(description, ".png");
    exportgraphics(gcf, fullfile(folder, filename));
    close(gcf);
end