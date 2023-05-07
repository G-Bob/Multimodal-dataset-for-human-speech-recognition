function [audio_sig_denoise, audio_spec_denoise, t, f] = spec_denoise(signal, fs_audio, threshold, ...
    frameLength, overlap, denoise)

% threshold = -40 by default

[Y, f, t] = stft(signal, fs_audio, "Window", hann(frameLength), "OverlapLength", overlap, ...
    "FFTLength", frameLength,"FrequencyRange",'onesided');
audio_spec_denoise = Y;

if denoise == "True"
    % imagesc(abs(YQZ));
    Y_ref = 20*log10(abs(Y));

%     figure;
%     subplot(211);
%     imagesc(t, f, Y_ref);colorbar;
%     xlabel('Time (s)');
%     ylabel('Frequency (Hz)');
%     title('Input signal spectrum')

    Y_ref(Y_ref < threshold) = 0; %Set denoise threshold of lower dB

    [i,j] = find(Y_ref==0);
    for m = 1:length(i)
        audio_spec_denoise(i(m),j(m)) = 0;
    end


    recovered_audio = istft(audio_spec_denoise, "Window", hann(frameLength), ...
        "OverlapLength", overlap, "FFTLength", frameLength,"FrequencyRange",'onesided');

    %audio_sig_denoise = recovered_audio * 3;
    audio_sig_denoise = recovered_audio * 1;
    % audiowrite(['denoise.wav'],audio_sig_denoise,fs_audio);
 
%     subplot(212);
%     imagesc(t, f, 20*log10(abs(audio_spec_denoise)));colorbar;
%     xlabel('Time (s)');
%     ylabel('Frequency (Hz)');
%     title('Input signal spectrum');
elseif denoise == "False"
    figure;
    audio_sig_denoise = signal;
    imagesc(t, f, 20*log10(abs(audio_spec_denoise)));colorbar;
end