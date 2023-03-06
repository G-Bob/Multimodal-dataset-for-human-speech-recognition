clear all;
close all;
clc;
%% Radar parameter（mmWave Studio default）
RP.c=3.0e8;
RP.B=768e6;       %Frequency modulation bandwidth
RP.K=30e12;       %Chirp slope
RP.T=RP.B/RP.K;         %Periodicity
RP.Tc=60e-6;     %chirp peoridicty
RP.fs=1e7;       %Sample rate
RP.f0=77e9;       %Initial Carrier Frequecy
RP.lambda=RP.c/RP.f0;   %Wavelength of radar
RP.d=RP.lambda/2;    %Distance of anteanna arra
RP.n_samples=512/2; %Sample rate/Chirp
% n_samples=128;
RP.N=512/2;         %Range FFT points
%n_chirps=1;  %Num of Chirps/frame
%n_chirps=64;
%M=1;         %Doppler FFT points
RP.n_RX=4;        %RX antennas number
RP.n_TX=1;        %TX antennas number
RP.n_frame = 65000;
RP.Q = 90;       %Angle FFT
% xx = 1;        %No.xx frame
RP.iterpoint = 1; % Set the interpolation points
RP.TX_index = 1;

fpath = ['D:\Multimodal_speech_dataset\Raw data_test\output'];
%% read in radar raw data and reshape

foldername = ['D:\Multimodal_speech_dataset\Raw data_test\mmWave'];
adcData = [];
NumofPeople = 1;
speechclass = ["vowel","word","sentences"];
Numofclass= 3; % loop for speechclass
classname = append(num2str(NumofPeople),'_ind-',speechclass(Numofclass));
classdir = append(foldername,'\',classname,'*.bin');
classdirlist = dir(classdir);
Numofind = 1:length(classdirlist); %loop for repetitions
keepvars = {'datafolder','speechclass','Numofclass','NumofPeople','classdirlist', 'Numofind','RP','foldername','fpath'};
clearvars('-except', keepvars{:});

%             for i=0:0
%                 fileName = [foldername,'\', ends, '._Raw_',num2str(i),'.bin'];
%                 %fileName = [foldername,ends,'.bin'];
%                 fid = fopen(fileName,'r');
%                 adcData_l = fread(fid, 'int16');
%                 adcData = [adcData; adcData_l];
%                 fclose(fid);
%             end

fileName = [classdirlist.folder,'\',classdirlist(Numofind).name];
disp(fileName);
%fileName = [foldername,ends,'.bin'];
fid = fopen(fileName,'r');
adcData = fread(fid, 'int16'); %16bits，
fclose(fid);

%% 2243+DCA1000
frames = reshape(adcData,[RP.n_RX,2,RP.n_samples,RP.n_TX,RP.n_frame]); % reshape radar signal
frames_complex = squeeze(frames(:,1,:,:,:) + 1j*frames(:,2,:,:,:));

%% Read timestamp
radlog_FN= [classdirlist.folder,'\','radarlog',classdirlist(Numofind).name(1:end-11),'.txt'];
fid = fopen(radlog_FN);
TiTime = textscan(fid,'%s',3,'delimiter','\n', 'headerlines',957);

Tistart = 10e-4*str2num(TiTime{1}{1}(11:13))+str2num(TiTime{1}{1}(8:9))+...
    str2num(TiTime{1}{1}(5:6))*60+str2num(TiTime{1}{1}(2:3))*3600;

range_win = hamming(RP.n_samples);   % Hamming window
data_match = [];
%% Extract phase infomation from specific range bin
separationInd = 10; % Separation to reduce the requirement of RAM
for repeatime = 1:separationInd
    range_profile = [];
    speed_profile = [];
    angle_profile = [];
    for storage_x = 1+(RP.n_frame)*((repeatime-1)/separationInd):RP.iterpoint:(RP.n_frame)*(repeatime/separationInd)
        xx = storage_x - (RP.n_frame)*((repeatime-1)/separationInd);
        for k=1:RP.n_RX
            temp=frames_complex(k,:,storage_x)'.*range_win;
            temp_fft=fft(temp,RP.N);    %Adopt N-point FFT for each chirp
            range_profile(:,k,xx)=temp_fft;
        end
        for n=1:RP.N   %range
            temp=range_profile(n,:,xx);
            temp_fft=fftshift(fft(temp,RP.Q));    % Adopt Q-point FFT for dopplerFFT result
            angle_profile(n,:,xx)=temp_fft;
        end
    end

    fs_frame = 1017; % Sample rate of frame
    Numframe = size(range_profile,4);
    Tstep = 1/fs_frame;
    t = 0:Tstep:Tstep*(Numframe-1);
    data_match(:,repeatime) = abs(squeeze(angle_profile(251,45,:))); 
    % The location is a predefined parameter from AoA-range map 
    
    plot(abs(squeeze(angle_profile(251,45,:)))');
    ylabel('Phase Differnce')
    xlabel('Time(s)')


end
%% Matching the piece of data.
dataSerial = reshape(data_match,[RP.n_frame,1]);
[s,w,t] = spectrogram(log10(abs(dataSerial)),512,500,512,fs_frame,'yaxis');
savename = strcat(fpath,'\',num2str(NumofPeople),'_',num2str(Numofclass),'_',num2str(Numofind),'.mat');
save(savename,'s','w','t','Tistart','TiTime');
