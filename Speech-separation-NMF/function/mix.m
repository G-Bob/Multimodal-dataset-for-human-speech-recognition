function [y_filt] = mix(audio_sig, laser_sig, fs_audio, fs_laser)


% Set the frame length and overlap
frameLength = 256;
overlap = 128;

% Resample the neck vibration signal to match the sample rate of the speech signal
resample_y = resample(laser_sig, fs_audio, fs_laser);

% Synchronize the start times of the two signals
delay = round(fs_audio * 0.1); % example delay of 100 ms
audio_sig = audio_sig(delay:end);
resample_y = resample_y(1:end-delay+1);

% Segment the speech signal and neck vibration signal into frames
frames_x = buffer(audio_sig, frameLength, overlap, 'nodelay');
frames_y = buffer(resample_y, frameLength, overlap, 'nodelay');

% Compute the power spectral density (PSD) of each frame
[psd_x, freq] = pwelch(frames_x, hann(frameLength), overlap, 2*frameLength, fs_audio);
[psd_y, freq] = pwelch(frames_y, hann(frameLength), overlap, 2*frameLength, fs_audio);

% % Plot the PSD
plot(freq, 10*log10(psd_x));
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Noisy Speech Signal PSD');
grid on;

% Estimate the noise PSD using the first few frames of the neck vibration signal
numNoiseFrames = 10;
noisePSD = mean(psd_y(:,1:numNoiseFrames), 2);

% Set the spectral floor
alpha = 2;
spectralFloor = alpha * noisePSD;

% Perform spectral subtraction for each frame
cleanPSD = zeros(size(psd_x));
for i = 1:size(psd_x,2)
    cleanPSD(:,i) = max(psd_x(:,i) - spectralFloor, 0);
end

% Reconstruct the enhanced speech signal
cleanFrames = sqrt(cleanPSD);
cleanSpeech = overlapAdd(cleanFrames, overlap);

% Compute the spectra of the original and enhanced signals
[psd_x_orig, freq_orig] = pwelch(audio_sig, hann(frameLength), overlap, 2*frameLength, fs_audio);
[psd_x_clean, freq_clean] = pwelch(cleanSpeech, hann(frameLength), overlap, 2*frameLength, fs_audio);

% Compute the spectra of the original and enhanced signals, and the noise floor
[psd_x_orig, freq_orig] = pwelch(audio_sig, hann(frameLength), overlap, 2*frameLength, fs_audio);
[psd_x_clean, freq_clean] = pwelch(cleanSpeech, hann(frameLength), overlap, 2*frameLength, fs_audio);

% Plot the spectra
figure;

% Plot the original speech spectrum, noise floor, and enhanced speech spectrum
subplot(3,1,1);
plot(freq_orig, 10*log10(psd_x_orig), 'b');
hold on;
plot(freq, 10*log10(spectralFloor), 'g');
plot(freq_clean, 10*log10(psd_x_clean), 'r');
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Original (blue) and Enhanced (red) Speech Spectra');
legend('Original', 'Noise Floor', 'Enhanced');
grid on;

% Plot the vibration spectrum
subplot(3,1,2);
plot(freq, 10*log10(psd_y), 'b');
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Vibration Signal Spectrum');
grid on;

% Plot the time-domain original and enhanced speech signals
subplot(3,1,3);
plot((0:length(audio_sig)-1)/fs_audio, audio_sig, 'b');
hold on;
plot((0:length(cleanSpeech)-1)/fs_audio, cleanSpeech, 'r');
xlabel('Time (s)');
ylabel('Power Spectral Density (dB/Hz)')
title('Enhaneced Signal Spectrum');
grid on;

reconSpeech = cleanSpeech/norm(cleanSpeech);

end

