function [audio_spect, t] = audio_read_npy(audioIndex, tf_type, fs_resample)

%% This script is used to test Spectrum sparation of single voice from gaussian noise with laser
% addpath(genpath('./'))
% Load voice file
% audiofile = "D:\dataset\spoeech_recognition\21-23\21_kinect_uwb\b1\audios\audio_0.wav"
[y, Fs] = audioread(audioIndex.filename);
% Pre-emphasis filter

preemph = [1, -0.97];
y = filter(preemph, 1, y);
audio_signal = y/norm(y);

% Time stamps
% [time_stamp] = json_read(audioIndex.tsfilename);

if tf_type == "fft"
    % Frame blocking
    frameSize = round(0.025*Fs); % 25 ms
    frameShift = round(0.010*Fs); % 10 ms
    frames = buffer(audio_signal, frameSize, frameSize-frameShift, 'nodelay');

    % Hamming window
    hamWin = hamming(frameSize);
    frames = bsxfun(@times, frames, hamWin);

    % FFT
    N = 2^nextpow2(frameSize);
    framesFFT = fft(frames, N);
    audio_spect = 20*log10(abs(framesFFT(1:N/2+1,:)));

    % Mel filterbank
    numFilters = 26;
    freqRange = [0, Fs/2];
    melFilters = melFilterBank(numFilters, N, Fs, freqRange);
    melSpect = melFilters*abs(framesFFT(1:N/2+1,:)).^2;

    % MFCCs
    numCoeffs = 12;
    mfccs = dct(log(melSpect));
    mfccs = mfccs(2:numCoeffs+1,:);



    % Time-frequency spectrogram
    % figure
    %t = (0:size(frames,2)-1)*(frameSize-frameShift)/Fs;
    t = (1:size(frames,2))*(frameShift/Fs);
    % imagesc(t, f, 20*log10(abs(framesFFT(1:N/2+1,:))))
    % axis xy
    % xlabel('Time (s)')
    % ylabel('Frequency (Hz)')
    % title('Time-Frequency Spectrogram')
    % colorbar


elseif tf_type == "direct"
    audio_spect = resample(audio_signal, fs_resample, Fs);
    t = 0:1/fs_resample:(size(audio_spect)-1)/fs_resample;
end

end