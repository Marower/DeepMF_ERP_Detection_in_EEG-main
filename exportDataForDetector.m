addpath (genpath('eeglab2022.1/'));
eeglab
load("dataset_ds005284.mat");

selected_channels = {'C3','C4','CZ','PZ'};

eeg_segments = cell(1, length(data));
for subi=1:length(data)
    EEG = data{subi};
    EEG = pop_select(EEG, 'channel', selected_channels);
    
    fs = 250;
    EEG = pop_resample(EEG, fs);
    
    EEG_signals = EEG.data;   % [4 × T]
    EEG_signals = zscore(EEG_signals, 0, 2);
    window_size = 2 * fs;  % 500 samples
    event_nr = length(EEG.event);
    
    % Select samples with events, with overlaping
    num_windows = 6;
    template_size = 200;
    step = floor((window_size - template_size) / num_windows);
    N = num_windows*event_nr;
    x_positive = zeros(N, 4, window_size);
    y_positive = zeros(N, 1, window_size);
    for eventi =1:event_nr
        latency = EEG.event(eventi).latency;
        start = latency-(window_size - template_size);
        for i = 1:num_windows
            
            idx_start = start + (i-1)*step + randi([0 step-1])+ 1;
            idx_end   = idx_start + window_size - 1;
            
            x_positive(i,:,:) = EEG_signals(:, idx_start:idx_end);
            y_positive(i,1,latency-start) = 1;
            if latency >= idx_start && latency <= idx_end
               
            else
                error('Event is outside the window');
            end
        end
    end

    signalLength = size(EEG.data, 2);   % total number of samples

    % Extract event latencies (rounded to integer samples)
    eventLatencies = round([EEG.event.latency]);
    eventLatencies = unique(eventLatencies);
    % Create mask
    eventMask = false(1, signalLength);
    eventMask(eventLatencies) = true;
    validStarts = [];

    for startIdx = 1:(signalLength - window_size + 1)
        
        if ~any(eventMask(startIdx:startIdx + window_size - 1))
            validStarts(end+1) = startIdx;
        end
        
    end
    if length(validStarts) < N
        error('Not enough event-free windows available.');
    end

    selectedStarts = validStarts(randperm(length(validStarts), N));
    x_negative = zeros(N,size(EEG.data,1), window_size);
    y_negative = zeros(N,1, window_size);
    for i = 1:N
        idx = selectedStarts(i);
        x_negative(i,:,:) = EEG.data(:, idx:idx+window_size-1);
    end
    X = cat(1, x_positive, x_negative);
    Y = cat(1, y_positive, y_negative);
    eeg_segments{subi}.X = X;
    eeg_segments{subi}.Y = Y;    
end
save('ERP_Detector_Dataset.mat', 'eeg_segments', '-v7.3');