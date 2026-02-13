addpath bva-io-master/
addpath (genpath('eeglab2022.1/'));
eeglab

datasetName = './ds005284/';
behav_data = tdfread([datasetName, 'participants.tsv']);
sub = behav_data.participant_id;

data = cell(1,size(sub, 1));

for subi = 1:size(sub, 1)
    subjectName = sub(subi,:);
    disp (subjectName)
    input_folder = [datasetName sub(subi, :) '/eeg/'];
    filename= [input_folder,sub(subi, :) '_task-26ByBiosemi_eeg.bdf'];
    EEG = pop_biosig(filename);
    %load events

    % Combine path and filename
    tableName= [input_folder,sub(subi, :) '_task-26ByBiosemi_events.tsv'];
    
    % Read the table
    events = readtable(tableName, ...
                       'FileType', 'text', ...
                       'Delimiter', '\t');
    % Keep only rows related to N-back
    isLaserEvent = contains(lower(events.value), 'condition 54');
    events = events(isLaserEvent, :);
    
    % Convert onset times (s) to sample indices
    eventSamples = round(events.onset * EEG.srate) + 1;
    
    EEG.event = [];  % clear existing events if needed
    
    for j = 1:length(eventSamples)
        EEG.event(j).latency = eventSamples(j);  % in samples
        EEG.event(j).type    = strcat (events.value{j}); % or a single label like 'nback'
    end
    % High-pass filter at 1 Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1);
    % Low-pass filter at 40 Hz
    EEG = pop_eegfiltnew(EEG, 'hicutoff', 40);
    
    EEG = pop_select( EEG, 'channel',{'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A27','A28','A29','A30','A31','A32','B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13','B14','B15','B16','B17','B18','B19','B20','B21','B22','B23','B24','B25','B26','B27','B28','B29','B30','B31','B32'});
    % channel location
    EEG=pop_chanedit(EEG, 'changefield',{1 'labels' 'FP1'},'changefield',{2 'labels' 'AF7'},'changefield',{3 'labels' 'AF3'},'changefield',{4 'labels' 'F1'},'changefield',{5 'labels' 'F3'},'changefield',{6 'labels' 'F5'},'changefield',{7 'labels' 'F7'},'changefield',{8 'labels' 'FT7'},'changefield',{9 'labels' 'FC5'},'changefield',{10 'labels' 'FC3'},'changefield',{11 'labels' 'FC1'},'changefield',{12 'labels' 'C1'},'changefield',{13 'labels' 'C3'},'changefield',{14 'labels' 'C5'},'changefield',{15 'labels' 'T7'},'changefield',{16 'labels' 'TP7'},'changefield',{17 'labels' 'CP5'},'changefield',{18 'labels' 'CP3'},'changefield',{19 'labels' 'CP3'},'changefield',{20 'labels' 'CP1'},'changefield',{21 'labels' 'P1'},'changefield',{22 'labels' 'P3'},'changefield',{23 'labels' 'P5'},'changefield',{24 'labels' 'P7'},'changefield',{19 'labels' 'CP1'},'changefield',{20 'labels' 'P1'},'changefield',{21 'labels' 'P3'},'changefield',{22 'labels' 'P5'},'changefield',{23 'labels' 'P7'},'changefield',{24 'labels' 'P9'},'changefield',{25 'labels' 'PO7'},'changefield',{26 'labels' 'PO3'},'changefield',{27 'labels' 'O1'},'changefield',{28 'labels' 'LZ'},'changefield',{29 'labels' 'OZ'},'changefield',{30 'labels' 'POZ'},'changefield',{31 'labels' 'PZ'},'changefield',{32 'labels' 'CPZ'},'changefield',{33 'labels' 'FPZ'},'changefield',{34 'labels' 'FP2'},'changefield',{35 'labels' 'AF8'},'changefield',{36 'labels' 'AF4'},'changefield',{37 'labels' 'AFZ'},'changefield',{38 'labels' 'FZ'},'changefield',{39 'labels' 'F2'},'changefield',{40 'labels' 'F4'},'changefield',{41 'labels' 'F6'},'changefield',{42 'labels' 'F8'},'changefield',{43 'labels' 'FT8'},'changefield',{44 'labels' 'FC6'},'changefield',{45 'labels' 'fc4'},'changefield',{45 'labels' 'FC4'},'changefield',{46 'labels' 'FC2'},'changefield',{47 'labels' 'FCZ'},'changefield',{48 'labels' 'CZ'},'changefield',{49 'labels' 'C2'},'changefield',{50 'labels' 'C4'},'changefield',{51 'labels' 'C6'},'changefield',{52 'labels' 'T8'},'changefield',{53 'labels' 'TP8'},'changefield',{54 'labels' 'CP6'},'changefield',{55 'labels' 'CP4'},'changefield',{56 'labels' 'CP2'},'changefield',{57 'labels' 'P2'},'changefield',{58 'labels' 'P4'},'changefield',{59 'labels' 'P6'},'changefield',{60 'labels' 'P8'},'changefield',{61 'labels' 'P10'},'changefield',{62 'labels' 'PO8'},'changefield',{63 'labels' 'PO4'},'changefield',{64 'labels' 'O2'});
    
    % average reference using EEG channels only
    EEG = pop_reref(EEG, {'P9','P10'});
    EEG = pop_runica(EEG, 'extended', 1, 'stop', 1e-6);
    
        
    EEG = eeg_checkset(EEG);
    data{subi} = EEG;
end
save ('dataset.mat',"data",'-v7.3')

%%
for i=1:size(sub, 1)
        data{i} = pop_resample(data{i}, 1000);
end
save ('dataset.mat',"data",'-v7.3')