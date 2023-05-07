function [laser_signal, f, t] = laser_read_npy(laserIndex, tf_type, fs_resample)
% This file is used for extract laser signal.
% laserfile = '.\dataset\laser_data';
% index_scene = 21;
% index_time = 1;
% Fs = 1470;
% dur = 65;

laserfile = laserIndex.filename;
%index_scene = laserIndex.scene;
%index_iter = laserIndex.iter;
% input_mode = laserIndex.input_mode;
Fs = laserIndex.Fs;
% dur = laserIndex.dur;
% t = 0:1/Fs:dur-1/Fs;

signal = readNPY(laserfile);

% % Extract sequence data
% addr = dir([laserfile,'\Laser_data_*',num2str(index_scene),'.mat']);
% addr_ts = [laserfile,'\person_',num2str(index_scene),'_timestamp.mat'];
% a = matfile([addr.folder,'\',addr.name]);
% 
% % Read time stamp
% time_stamp(1,:) = read_ts_laser(addr_ts, index_iter, "starttime");
% time_stamp(2,:) = read_ts_laser(addr_ts, index_iter, "stoptime");
% x = a.lip_dataset_Y(index_iter,:) + 1j * a.lip_dataset_X(index_iter,:);
%time_stamp(1,:) = a.datatimestart(index_iter,:);
%time_stamp(2,:) = a.datatimestop(index_iter,:);
% % Generate complex signal
% x = exp(1i*2*pi*100*t) + exp(1i*2*pi*200*t);  % Example signal

% Save signal as WAV file
% audiowrite('signal.wav', signal, Fs);

% Preprocess signal
frameSize = 512;
frameShift = frameSize/2;

%% Use only one-dimension signal or two-dimension signal
% if input_mode == "real"'
%     signal = real(x);
% elseif input_mode == "abs"
%     signal = abs(x);
% elseif input_mode == "imag"
%     signal = imag(x);
% else
%     error('INPUT MODE MUST BE SELECTED AMONG [read], [abs] or [imag]')
% end
signal = signal/norm(signal);

if tf_type == "fft"
    %frames = buffer(x, frameSize, frameSize-frameShift, 'nodelay');
    frames = buffer(signal, frameSize, frameSize-frameShift, 'nodelay');
    hamWin = hamming(frameSize);
    frames = bsxfun(@times, frames, hamWin);

    % FFT
    N = frameSize;
    framesFFT = fft(frames, N, 1);
    laser_signal = 20*log10(abs(framesFFT(1:N/2+1,:)));

    % Time-frequency spectrogram
    freqRange = [0, Fs/2];
    f = freqRange(1):Fs/N:freqRange(2);
    t = (0:size(frames,2)-1)*(frameSize-frameShift)/Fs;
    % imagesc(t, f, 20*log10(abs(framesFFT(1:N/2+1,:))))
    % axis xy
    % xlabel('Time (s)')
    % ylabel('Frequency (Hz)')
    % title('Time-Frequency Spectrogram')
    % colorbar
elseif tf_type == "direct"
    laser_signal = resample(signal, fs_resample, Fs);
    f = nan;
    t = 0:1/fs_resample:(size(laser_signal,1)-1)/fs_resample;
end
% Play signal
% sound(signal, Fs);

end

