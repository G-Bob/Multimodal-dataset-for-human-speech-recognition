addpath('D:\mouth data\Code\npy-matlab-master\npy-matlab') % add the path of npy reader

figure(1);
uwbfilefolfer = ('D:\mouth data\Processed_cut_data\uwb_processed\');
mmwavefolfer = ('D:\mouth data\Processed_cut_data\radar_processed\');
laserfolder = ('D:\mouth data\Processed_cut_data\laser_processed\');
audiofolder = ('D:\mouth data\Processed_cut_data\kinect_processed\');

%% Change the user index and sample index
user = '1'; % only for 1,7,14
types = 'sentences'; % vowel, word, sentences
type_index = '1';
sample_index = '1';
%%
subplot(4,1,4) % Voice
[fourth_spectrum,fs] = audioread([audiofolder,user,'\',types,type_index,'\audios\',...
    'audio_proc_',num2str(str2num(sample_index)-1),'.wav']); % wav index is one less than radar

pspectrum(fourth_spectrum,fs,'spectrogram','OverlapPercent',0.9,'Leakage',1,'MinThreshold',-60,'TimeResolution', 10e-3)
axis xy; set(gca,'FontSize',14);
title('Audio signal');
ylim([0 1.2]);
timelength = length(fourth_spectrum)/fs; % Time length cut by Kinect

subplot(4,1,1) % UWB radar
uwbaddress = dir([uwbfilefolfer,user,'\',types,'_',type_index,'\',user,'*','1_sample',sample_index,'.npy']);

first_spectrum = readNPY([uwbaddress.folder,'\',uwbaddress.name]);

fre_bandwidth_uwb = 30; % Cropped spectrum pre-requsite
y_doppler_uwb = linspace(-fre_bandwidth_uwb,fre_bandwidth_uwb,size(first_spectrum,1));
x_time_uwb = linspace(0,timelength,size(first_spectrum,2));

x_time_uwb_test = linspace(0,136,size(first_spectrum,2));

imagesc(x_time_uwb,y_doppler_uwb,20*log10(first_spectrum./max(first_spectrum)));
% imagesc(x_time_uwb_test,y_doppler_uwb,20*log10(first_spectrum./max(first_spectrum)));

axis xy; set(gca,'FontSize',14);
colormap('jet');
title('UWB radar signal');
xlabel('Time (s)','FontSize',14); ylabel('Doppler (Hz)','FontSize',14);caxis([-10 0]);%colorbar;
%ylim([-20 20])   %% Cut of Spectrograms from -30 Hz to 30 HZ


subplot(4,1,2) % FMCW radar
fmcwaddress = dir([mmwavefolfer,user,'\',types,type_index,'\','*sample',sample_index,'.npy']);

second_spectrum = readNPY([fmcwaddress.folder,'\',fmcwaddress.name]);

fre_bandwidth_mmwave = 509; % Cropped spectrum pre-requsite
y_doppler_mmwave = linspace(0,fre_bandwidth_mmwave,size(second_spectrum,1));
x_time_mmwave = linspace(0,timelength,size(second_spectrum,2));

imagesc(x_time_mmwave,y_doppler_mmwave,20*log10(second_spectrum./max(second_spectrum)));
axis xy; set(gca,'FontSize',14);
colormap('jet');
title('FMCW radar signal');
xlabel('Time (s)','FontSize',14); ylabel('Doppler (Hz)','FontSize',14);caxis([-70 0]);
ylim([0 100])   %% Cut of Spectrograms from -30 Hz to 30 HZ

subplot(4,1,3) % Laser
laserAddress = dir([laserfolder,user,'\',types,type_index,'\','sample*',sample_index,'.npy']);
third_spectrum = readNPY([laserAddress.folder,'\sample',type_index,'.npy']);

x_time_laser = linspace(0,timelength,size(third_spectrum,1));

plot(x_time_laser,third_spectrum)
axis xy; set(gca,'FontSize',14);
title('Laser signal');
xlabel('Time (s)','FontSize',14); ylabel('Magnitude','FontSize',14);
xlim([0 timelength])
