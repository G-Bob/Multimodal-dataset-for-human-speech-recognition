function [scd,srd,t_crop_audio] = nmf_mix(audio_sig, laser_sig, fs_audio, t_audio)

threshold = -50; % Denoise threshold. 

% Set the frame length and overlap
frameLength = 512;
overlap = 256;
denoise = "True"; % Try to find renonsence part of audio


%% Transfer to STFT and (Denoise optional)  

[audio_sig_denoise, audio_spec_denoise, t_audio_spec, f_audio_spec] = spec_denoise(audio_sig, fs_audio, threshold, frameLength, overlap, denoise);
    
% laser data is purer than audio and may not need filter:
    
[laser_sig_denoise, laser_spec_denoise, t_laser_spec, f_laser_spec] = spec_denoise(laser_sig, fs_audio, threshold, frameLength, overlap, denoise);

%% decompose the audio spec to two lower dimension multiplication
basis_num = 8; % default is 20
max_iter_GD = 50; % default is 500
learning_rate = 1.0e-8; % default is 1e-10

asd = abs(audio_spec_denoise);
[W,H] = NMF_GD(asd, basis_num, max_iter_GD, learning_rate);

%% Get reference matrix
lsd = abs(laser_spec_denoise);
[WB,HB] = NMF_GD(lsd,basis_num, max_iter_GD, learning_rate); % Gradient descent

%% cluster training
[Wm,Wc,Wr,Hm,Hc,Hr] = clusterH(W,WB,H,basis_num);
for i=1:100
    if size(Wm,2)<1
        % If size(Wm,2) == 0, the separation finished
        [Sc,Sr] = reconstruction(audio_spec_denoise,Wc,Wr,Hc,Hr);
        break
    else
        % Continue to separate Sr,Sc from Wm
        [Wm,Wc,Wr,Hm,Hc,Hr] = clusterM(Wm,Wc,Wr,Hm,Hc,Hr);
        %fprintf('k-means finished')
    end
end
sc = istft(Sc,"Window", hann(frameLength), "OverlapLength", overlap, "FFTLength", frameLength,"FrequencyRange",'onesided');


scd = sc;
audiowrite(['filtered1.wav'],scd,fs_audio);
sr = istft(Sr,"Window", hann(frameLength), "OverlapLength", overlap, "FFTLength", frameLength,"FrequencyRange",'onesided'); 
srd = sr;
audiowrite(['filtered2.wav'],srd,fs_audio);
%% Plot in time and frequency
figure(2);
subplot(221);
plot(t_audio, audio_sig);
title('original');
xlabel('Samples');
ylabel('Amplitudes');
grid;

subplot(222);
plot(t_audio(1:size(sc,1)),audio_sig_denoise(1:size(sc,1)));
ylim([min(audio_sig) max(audio_sig)]);
title('Denoised audio signal')
xlabel('Time(s)')
ylabel('Amplitude')
grid;

subplot(223);
plot(t_audio(1:size(sc,1)),audio_sig_denoise(1:size(sc,1)));
hold on;
plot(t_audio(1:size(sc,1)), scd);
hold off
%xlim([0 t_audio_spec(size(sc,1))]);
ylim([min(audio_sig) max(audio_sig)]);
title('Masked audio signal')
xlabel('Time(s)')
ylabel('Amplitude')
grid;

t_crop_audio = t_audio(1:size(sr,1));

subplot(224);
plot(t_crop_audio,audio_sig_denoise(1:size(sr,1)));
hold on;
plot(t_crop_audio, srd);
hold off
xlim([0 t_crop_audio(end)]);
ylim([min(audio_sig) max(audio_sig)]);
title('Recovered audio signal')
xlabel('Time(s)')
ylabel('Amplitude')
grid;


figure(3);
subplot(221)
title('original laser signal spectrum')
imagesc(t_laser_spec, f_laser_spec, 20*log10(abs(laser_spec_denoise)));colorbar;

subplot(222)
title('original audio signal spectrum')
imagesc(t_audio_spec, f_audio_spec, 20*log10(abs(audio_spec_denoise)));colorbar;

subplot(223)
title('Masked signal spectrum')
imagesc(t_audio_spec(1:size(Sc,2)), f_audio_spec, 20*log10(abs(Sc)));colorbar;
subplot(224)
title('Unmasked signal spectrum')
imagesc(t_audio_spec(1:size(Sc,2)), f_audio_spec, 20*log10(abs(Sr)));colorbar;

disp('Finished')

end