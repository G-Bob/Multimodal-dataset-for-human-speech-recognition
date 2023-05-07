function [SDR1, SDR2, SIR1, SIR2] = nmf_mask(ParSetting)
% clear;
% clc;
% close all;
% 
% addpath(genpath('./function/'))
% addpath(genpath('./dependency/'))
% 
% cmd_type = "direct";
% mix_type = "nmf";
% id_audio = '18';
% ind_audio = '2';
% id_laser = '17';
% ind_laser = '2';
% fs_resample = 1470;
% delay = 1; % Delay to separate two sources in time domain

cmd_type = ParSetting.cmd_type;
mix_type = ParSetting.mix_type;
id_audio = ParSetting.id_audio;
ind_audio = ParSetting.ind_audio;
id_laser = ParSetting.id_laser;
ind_laser = ParSetting.ind_laser;
fs_resample = ParSetting.fs_resample;
delay = ParSetting.delay; % Delay to separate two sources in time domain
%% Read laser signal
% laserIndex.scene = str2double(id_laser);
% laserIndex.iter = str2double(ind_laser) - 1;
laserIndex.Fs = 1470;
laserIndex.filename =  ['.\dataset\',id_laser,'\sentences1\Laser_data_person_sample',...
    ind_laser,'.npy'];
% laserIndex.input_mode = "imag";
[laser_sig, f_laser, t_laser] = laser_read_npy(laserIndex, cmd_type, fs_resample);

%% Read audio signal related to laser's index
% audioIndexA.scene = str2double(id_laser);
audioIndexA.iter = str2double(ind_audio) - 1;
audioIndexA.filename = ['.\dataset\',id_laser,'\audios\audio_proc_',num2str(audioIndexA.iter),'.wav'];
[audio_sigA, t_audioA] = audio_read_npy(audioIndexA,cmd_type,fs_resample); % Transfer to 16k

%% Read filtered audio spectrum
% audioIndexB.scene = str2double(id_audio);
audioIndexB.iter = str2double(ind_audio) - 1;
audioIndexB.filename = ['.\dataset\',id_audio,'\audios\audio_proc_',num2str(audioIndexB.iter),'.wav'];
% audioIndex.tsfilename = ['.\21-23\',num2str(audioIndex.scene),'_kinect_uwb\b',num2str(audioIndex.iter),'\timestamps\timestamp_0.json'];
[audio_sigB, t_audioB] = audio_read_npy(audioIndexB,cmd_type,fs_resample);

%% Crop the overlap sequence

minimum_time = time_allocation_singleside(t_laser,t_audioA,t_audioB);
[cropped_seq] = ...
    crop_mintime_sequence(t_laser, t_audioA, t_audioB, minimum_time);

laser_sig_sync = laser_sig(1:cropped_seq);
audio_sigA_sync = audio_sigA(1:cropped_seq);
audio_sigB_sync = audio_sigB(1:cropped_seq);

t_audio = t_audioA(1:cropped_seq);
t_laser = t_laser(1:cropped_seq);
%% Circshift the second audio signal to simulate time delay

audio_sigB_sync = circshift(audio_sigB_sync, delay*fs_resample);


%% Voice addition
mix = audio_sigA_sync + audio_sigB_sync; % No delay

%% Plot laser and audio spectrum
figure(1)
subplot(221)
plot(t_audio, audio_sigA_sync);
xlim([0 t_audio(end)]);
title('Audio signal of Subject A')
xlabel('Time(s)')
ylabel('Amplitude')

subplot(222)
plot(t_laser, laser_sig_sync);
xlim([0 t_laser(end)]);
title('Laser signal of Subject A')
xlabel('Time(s)')
ylabel('Amplitude')
subplot(223)
plot(t_audio, audio_sigB_sync);
xlim([0 t_audio(end)]);
title('Audio signal of Subject B')
xlabel('Time(s)')
ylabel('Amplitude')
subplot(224)
plot(t_audio, mix);
xlim([0 t_audio(end)]);
title('Integration audio signal of Subject A and B')
xlabel('Time(s)')
ylabel('Amplitude')


% figure(1)
% imagesc(t_laser, f_laser, laser_spec)
% axis xy
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Time-Frequency Spectrogram')
% xlim([cropped_seq1(1) 700]);
% colorbar
%
% figure(2)
% imagesc(t_audio, f_audio, audio_spec)
% axis xy
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Time-Frequency Spectrogram')
% ylim([0 700]);
% colorbar
%% Apply spectrum substraction
if cmd_type == "direct"
    if mix_type == "nmf"
        [maskedVoice, restVoice, t_crop_audio] = nmf_mix(mix, laser_sig_sync, fs_resample, t_audio);
        [audio_sigA_denoise, ~, ~, ~] = spec_denoise(audio_sigA_sync, fs_resample, -40, 512, 256, "True");
        [audio_sigB_denoise, ~, ~, ~] = spec_denoise(audio_sigB_sync, fs_resample, -40, 512, 256, "True");

    elseif mix_type == "substraction"
        maskedVoice = mix(audio_spec_sync, laser_spec_sync, fs_audio, laserIndex.Fs);
    end
elseif cmd_type == "fft_plot"

%% Plot sync spectrums
    figure(1)
    imagesc(t_laser_sync, f_laser, laser_spec_sync)
    axis xy
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title('Time-Frequency Spectrogram')
    %xlim([cropped_seq1(1) 700]);
    colorbar
    
    figure(2)
    imagesc(t_audio_sync, f_audio, audio_spec_sync)
    axis xy
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title('Time-Frequency Spectrogram')
    %ylim([0 700]);
    colorbar
end
diff_len = (size(audio_sigA_sync,1) - size(maskedVoice,1))/2; %
% audio_sigA_sync_crop = audio_sigA_sync(diff_len+1:size(audio_sigA_sync,1)-diff_len);
% audio_sigB_sync_crop = audio_sigB_sync(diff_len+1:size(audio_sigB_sync,1)-diff_len);
audio_sigA_sync_crop = audio_sigA_sync(diff_len*2+1:size(audio_sigA_sync,1));
audio_sigB_sync_crop = audio_sigB_sync(diff_len*2+1:size(audio_sigB_sync,1));

figure(4);
subplot(221);
plot(t_crop_audio, audio_sigB_denoise);
title('original');
xlabel('Samples');
ylabel('Amplitudes');
ylim([min(mix) max(mix)]);xlim([0 t_crop_audio(end)]);
grid;

subplot(222);
plot(t_crop_audio,audio_sigA_denoise);
ylim([min(mix) max(mix)]);xlim([0 t_crop_audio(end)]);
title('Denoised audio signal')
xlabel('Time(s)')
ylabel('Amplitude')
grid;

subplot(223);
plot(t_crop_audio, audio_sigA_denoise + audio_sigB_denoise);
hold on;
plot(t_crop_audio, maskedVoice);
hold off
xlim([0 t_crop_audio(end)]);
ylim([min(mix) max(mix)]);
title('Masked audio signal')
xlabel('Time(s)')
ylabel('Amplitude')
grid;

subplot(224);
plot(t_crop_audio, audio_sigA_denoise + audio_sigB_denoise);
hold on;
plot(t_crop_audio, restVoice);
hold off
xlim([0 t_crop_audio(end)]);
ylim([min(mix) max(mix)]);
title('Recovered audio signal')
xlabel('Time(s)')
ylabel('Amplitude')
grid;

%% Calculate the SDR
[SDR1, SIR1] = evaluate_separation(maskedVoice, audio_sigB_denoise, 1e-6);
[SDR2, SIR2] = evaluate_separation(restVoice, audio_sigA_denoise, 1e-6);

disp(['SDR1: ', num2str(SDR1), ', SIR1: ', num2str(SIR1)]);
disp(['SDR2: ', num2str(SDR2), ', SIR2: ', num2str(SIR2)]);

end
%% Play signal
% sound(maskedVoice, f_laser);