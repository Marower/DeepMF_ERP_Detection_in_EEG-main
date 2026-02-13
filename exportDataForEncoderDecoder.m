addpath (genpath('eeglab2022.1/'));
eeglab
load("dataset_ds005284.mat");
all_windows = cell(length(data),1); 
selected_channels = {'C3','C4','CZ','PZ'};
for subi=1:length(data)
    EEG = data{subi};
    EEG = pop_select(EEG, 'channel', selected_channels);
    
    fs = 250;
    EEG = pop_resample(EEG, fs);
    

    EEG_signals = EEG.data;   % [4 × T]
    EEG_signals = zscore(EEG_signals, 0, 2);
    T = size(EEG_signals,2);
    

    window_size = 2 * fs;  % 500 samples

    overlap = 0.80;
    stride = round(window_size * (1 - overlap));
    
    num_windows = floor((T - window_size) / stride) + 1;
    disp(num_windows)
    
    windows = zeros(4, window_size, num_windows);
    
    for i = 1:num_windows
        
        idx_start = (i-1)*stride + 1;
        idx_end   = idx_start + window_size - 1;
        
        windows(:,:,i) = EEG_signals(:, idx_start:idx_end);
        
    end

    windows = permute(windows, [3 1 2]);
    all_windows{subi} = windows;
end
eeg_segments = cat(1, all_windows{:});
save('eeg_dataset.mat', 'eeg_segments', '-v7.3');