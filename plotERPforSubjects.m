addpath bva-io-master/
addpath (genpath('eeglab2022.1/'));
eeglab
load("dataset_ds005284.mat");

folder = 'EvokedPotentialPlots';
htmlGray = [128 128 128]/255;

for i = 1:length(data)
    disp (num2str(i))
    EEG = data{i};
    EEG = pop_resample(EEG, 1000);
    allTypes = unique({data{1}.event.type});
    for t = 1:length(allTypes)
        eventType = allTypes{t};
        EEG_ep = pop_epoch(EEG, eventType, [-0.20 0.8]);
        % baseline-correct
        EEG_ep = pop_rmbase(EEG_ep, [-200 0]);
        EEG_ep = pop_autorej(EEG_ep, 'threshold', 100, 'startprob', 5, 'maxrej', 5, 'nogui', 'on');
    

        t = EEG_ep.times;
        s = size(EEG_ep.data);
        channelLabels = {EEG_ep.chanlocs.labels};
        gcf = figure();
        hold on
        ERP = mean(EEG_ep.data, 3);

        for chan = 1:length(channelLabels)
            plot(t, ERP(chan, :), 'LineWidth', 0.5, 'Color', htmlGray)
        end
        xlabel('Time (ms)')
        ylabel('Amplitude (µV)')
        description = strcat(strcat("Participant_", num2str(i)),...
                            strcat("_event_", eventType));
            
        grid on
        filename = strcat(description, ".png");
        set(gca,'FontSize',14)
        exportgraphics(gcf, fullfile(folder, filename));
        close(gcf);
    end
end