addpath bva-io-master/
addpath (genpath('eeglab2022.1/'));
eeglab
load("dataset_ds005284.mat");

selected_channels = {'C3','C4','CZ','PZ'};

aveERP = zeros(length(selected_channels),300);   
for figi=1:length(selected_channels)
    for i=1:size(data,2)
        disp (num2str(i))
        EEG = data{i};
        EEG = pop_select(EEG, 'channel', selected_channels);
        fs = 250;
        EEG = pop_resample(EEG, fs);

        EEG_ep = pop_epoch(EEG, 'condition 54', [-0.20 1.0]);
        % baseline-correct
        EEG_ep = pop_rmbase(EEG_ep, [-200 0]);   
        EEG_ep = pop_autorej(EEG_ep, 'threshold', 100, 'startprob', 5, 'maxrej', 5, 'nogui', 'on');        
        ERP = mean(EEG_ep.data, 3);
        averagedERP(i,:) = ERP(figi, :);

    end
    aveERP(figi,:) = mean(averagedERP, 1);
end


templates = aveERP(:,50:249);
templates(:,1:30) = 0;
templates(:,190:200) = 0;
templates = smoothdata(templates, 'gaussian', 20);
%Flip templates for Matched Filter
templates = fliplr(templates);
save('channels_templates.mat', 'templates', '-v7.3');
for i =1:4
    subplot(2,2,i)
    plot(templates(i,:))
end