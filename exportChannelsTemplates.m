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
% Initialization of ReLU
for i =1:4
    templates(i,:) = templates(i,:) - mean(templates(i,:));
    templates(i,:) = templates(i,:) ./ std(templates(i,:));
   
end
use_He = false;
file_name = 'channels_templates';
if use_He
 % "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"

 % Step 3: He scaling
    [kh, kw, Cin] = size(templates);
    
    fan_in = kh * kw * Cin;
    scale = sqrt(2 / fan_in);
    
    templates = templates * scale;
    file_name = strcat(file_name,'_He');
end
file_name = strcat(file_name,'.mat');
%Flip templates for Matched Filter
templates = fliplr(templates);
save(file_name, 'templates', '-v7.3');


for i =1:4
    subplot(2,2,i)
    plot(templates(i,:))
end