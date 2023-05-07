clear;
clc;
close all;

addpath(genpath('./function/'))
addpath(genpath('./dependency/'))

ParSetting.cmd_type = "direct";
ParSetting.mix_type = "nmf";
ParSetting.fs_resample = 1470;
resultMatrix = [];

for i = 1:7
    ParSetting.id_audio = '18';
    ParSetting.ind_audio = num2str(i); % Storage of audio signal are named from 0 to 6
    ParSetting.id_laser = '17';
    ParSetting.ind_laser = num2str(i);
    ParSetting.delay = 1; % Delay to separate two sources in time domain
    [SDR1, SDR2, SIR1, SIR2] = nmf_mask(ParSetting);
    resultMatrix = [resultMatrix; [SDR1, SDR2, SIR1, SIR2]];
    disp(['Complete separation of ',ParSetting.id_audio, ' in ',ParSetting.ind_audio, ...
        ' with ', ParSetting.id_laser, ' of ', ParSetting.ind_laser]);
end

avg_sdr1 = mean(resultMatrix(:,1));
avg_sdr2 = mean(resultMatrix(:,2));
avg_sir1 = mean(resultMatrix(:,3));
avg_sir2 = mean(resultMatrix(:,4));

disp(['Overall, the average SDR of masked audio is ', num2str(avg_sdr1), newline,...
    'average SIR of masked audio is ', num2str(avg_sir1),newline,...
    'average SDR of recovered audio is ', num2str(avg_sdr2),newline,...
    'average SIR of recovered audio is ', num2str(avg_sir2)])