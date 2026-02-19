load("LOO_results.mat")

F1_deepMF(size(results)) = 0;
F1_deepMF_He(size(results)) = 0;
F1_rand(size(results)) = 0;

for i = 1:length(results)
    TP = results{i}.tp_deepMF;
    FN = results{i}.fn_deepMF;
    FP = results{i}.fp_deepMF;
    TP = double(TP);
    FP = double(FP);
    FN = double(FN);
    Sensitivity = TP ./ (TP + FN);
    Precision   = TP ./ (TP + FP);
    F1_deepMF(i) = 2 * (Precision .* Sensitivity) ./ (Precision + Sensitivity);

    TP = results{i}.tp_deepMF_He;
    FN = results{i}.fn_deepMF_He;
    FP = results{i}.fp_deepMF_He;
    TP = double(TP);
    FP = double(FP);
    FN = double(FN);
    Sensitivity = TP ./ (TP + FN);
    Precision   = TP ./ (TP + FP);
    F1_deepMF_He(i) = 2 * (Precision .* Sensitivity) ./ (Precision + Sensitivity);

    TP = results{i}.tp_rand;
    FN = results{i}.fn_rand;
    FP = results{i}.fp_rand;
    TP = double(TP);
    FP = double(FP);
    FN = double(FN);
    Sensitivity = TP ./ (TP + FN);
    Precision   = TP ./ (TP + FP);
    F1_rand(i) = 2 * (Precision .* Sensitivity) ./ (Precision + Sensitivity);
end
%%
F1_all = [F1_rand(:), F1_deepMF_He(:)];

figure;
b = bar(F1_all, 'grouped');

xlabel('Participant')
ylabel('F1 Score')
title('LOO Performance per Participant')
ylim([0 1])
grid on
m = mean(F1_all);
label1 = strcat('Standard Initialization, mean(F1) = ', num2str(m(1),'%.2f')); 
label2 = strcat('Deep-MF, mean(F1) = ', num2str(m(2),'%.2f')); 
legend({label1,label2}, 'Location','best')