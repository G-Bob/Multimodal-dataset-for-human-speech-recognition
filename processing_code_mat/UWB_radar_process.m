clear all
close all
clc
PRF=300; %300 frames per second
datafolder = ['D:\Multimodal_speech_dataset\Raw data_test\UWB'];
outputfilefolder = ['D:\Multimodal_speech_dataset\Raw data_test\output'];

NumofPeople = 1;
%loop for number if subjects

classdirlist = dir(datafolder);
for Numofind=3:length(classdirlist) %loop for repetitions
    keepvars = {'datafolder','speechclass','Numofclass','NumofPeople','classdirlist', 'Numofind','PRF','outputfilefolder'};
    clearvars('-except', keepvars{:});

    file = strcat(classdirlist(Numofind).folder,'\');
    timestampfile = dir([file,'\timestamp_ind','*']);
    % Print dataset faults
    if size(dir(file),1) ~= 4
        disp(['Here is missing xethru file in: ',file])
    end
    if timestampfile(1).name(end-1) == '.'
        newname = append([timestampfile(1).folder,'\',timestampfile(1).name,'at']); % Rename from .m to .mat
        movefile([timestampfile(1).folder,'\',timestampfile(1).name], newname);
        load(newname);
    else
        load([timestampfile(1).folder,'\',timestampfile(1).name]);
    end

    datfolder = dir([file,'\xethru_recording_','*\']);
    datfile = strcat([datfolder(3).folder,'\',datfolder(3).name]);
    fid=fopen(datfile,'rb');
    f=dir(datfile);
    fsize=f.bytes;

    %The WHILE loop reads the file data until End of File is reached and saves the data
    %into Datastream variable
    ctr = 0;
    hdrMat = [];
    FrameMat = [];
    TimeVec = [];

    while(1)
        ContentId=fread(fid,1,'uint32');
        Info=fread(fid,1,'uint32');
        ctr=ctr+1;
        Length=fread(fid,1,'uint32');
        Data=fread(fid,182,'float');
        if feof(fid)
            break
        end
        Datastream(:,ctr)=Data;
        data_length(ctr)=Length;
    end


    %% Now we need to map Datastream into a complex range-time-intensity matrix
    %we do that and plot the RTI
    %We perform MTI and plot the result

    for n=1:size(Datastream,2)
        Data=Datastream(:,n);
        Data_show = Data(1:end/2) + 1i*Data(end/2 + 1:end);
        i_vec = Data(1:data_length(n)/2);
        q_vec = Data(data_length(n)/2+1:data_length(n));
        iq_vec = i_vec + 1i*q_vec;
        RTI_Matrix(:,n)=iq_vec;
    end

    Data_amplitude=abs(RTI_Matrix');
    Data_phase=angle(RTI_Matrix');
    Data_IQ=RTI_Matrix';

    %Define range axis and time axis
    frame_start=0.3858; %these 2 parameters decribe the initial and final distance of the first and last range bin - again consider whether to put them at the beginning of the script
    frame_stop=5.0154;
    %Generate range vector or range axis
    bin_length = 8 * 1.5e8/23.328e9; % range_decimation_factor * (c/2) / fs.
    range_vector = (frame_start-1e-5):bin_length:(frame_stop+1e-5); % +-1e-5 to account for float precision.
    RangeAxis=range_vector;
    TimeAxis=[1:size(RTI_Matrix,2)]/PRF;

    %                 figure;
    %                 imagesc(RangeAxis,TimeAxis,20*log10(abs(Data_IQ./max(Data_IQ(:)))));
    %                 colormap('jet');
    %                 set(gca,'FontSize',14);
    %                 title('RTI - No MTI')
    %                 xlabel('Range [m] ','FontSize',14);
    %                 ylabel('Time [s]','FontSize',14);
    %                 caxis([-40 0]);
    %                 colorbar;
    %
    Data_RTI_MTIAdap=Data_IQ-( 0.95*circshift(Data_IQ,[1,0]) + 0.05*Data_IQ ); %Application of MTI

    %                 figure;
    %                 imagesc([1:size(Data_IQ,2)],TimeAxis,20*log10(abs(Data_RTI_MTIAdap./max(Data_RTI_MTIAdap(:)))));
    %                 colormap('jet');
    %                 set(gca,'FontSize',14);
    %                 title('RTI - With MTI')
    %                 xlabel('Range bins ','FontSize',14);
    %                 ylabel('Time [s]','FontSize',14);
    %                 caxis([-40 0]);
    %                 colorbar;

    %% Range Time Plot

    %                 figure;
    %                 imagesc(RangeAxis,TimeAxis,20*log10(abs(Data_RTI_MTIAdap./max(Data_RTI_MTIAdap(:)))));
    %                 colormap('jet');
    %                 set(gca,'FontSize',14);
    %                 title('RTI - With MTI')
    %                 xlabel('Range [m] ','FontSize',14);
    %                 ylabel('Time [s]','FontSize',14);
    %                 caxis([-40 0]);
    %                 colorbar;

    %% Create Range-Doppler map

    Data_RD=fftshift(fft(Data_IQ),1);
    Data_RD_MTIAdap=fftshift(fft(Data_RTI_MTIAdap),1);
    DopplerAxis=linspace(-PRF/2,PRF/2,size(Data_RD,1));

    %                 figure;
    %                 imagesc([1:size(Data_RD,2)],DopplerAxis,20*log10(abs(Data_RD./max(Data_RD(:)))));
    %                 colormap('jet'); set(gca,'FontSize',14); title('RD - Without MTI');
    %                 axis xy
    %                 xlabel('Range bins ','FontSize',14); ylabel('Doppler bins','FontSize',14);caxis([-60 0]);colorbar;
    %
    %                 figure;
    %                 imagesc([1:size(Data_RD,2)],DopplerAxis,20*log10(abs(Data_RD_MTIAdap./max(Data_RD_MTIAdap(:)))));
    %                 colormap('jet'); set(gca,'FontSize',14); title('RD - With MTI');
    %                 axis xy
    %                 xlabel('Range bins ','FontSize',14); ylabel('Doppler bins','FontSize',14);caxis([-30 0]);colorbar;

    %% Second MTI filter applied as a Butterworth 4th order - is it needed? Dunno check
    [b,a]=butter(4, 0.01, 'high');
    [h,fl]=freqz(b,a,size(Data_RTI_MTIAdap,2));
    for k=1:size(Data_RTI_MTIAdap,1)
        Data_RTI_complex_MTIFilt(k,:)=filter(b,a,Data_RTI_MTIAdap(k,:));
    end

    %% Create Spectrogram -
    %Micro-Doppler parameters: the working variable AND the 3 basic parameters:
    %window length, overlap percentage, and FFT padding factor

    Data_ForMicroDop=Data_RTI_complex_MTIFilt; %Data_ForMicroDop is our working variable

    TimeWindowLength =128; %Bear in mind that the window length is measured in samples, 128 samples
    OverlapFactor = 0.95;
    OverlapLength = round(TimeWindowLength*OverlapFactor);
    Pad_Factor = 16;

    %These are just working parameters that are given once you define those
    %above
    FFTPoints = Pad_Factor*TimeWindowLength;
    DopplerBin=PRF/(FFTPoints);
    DopplerAxisMD=-PRF/2:DopplerBin:PRF/2-DopplerBin;
    WholeDuration=size(Data_ForMicroDop,1)/PRF;
    NumSegments=floor((size(Data_ForMicroDop,1)-TimeWindowLength)/floor(TimeWindowLength*(1-OverlapFactor)));
    TimeAxisMD=linspace(0,WholeDuration);

    %%Select the range bins over which we want to calculate the micro-Doppler -
    %%consider whether you want to move these variables at beginning of code
    BinStart=1;
    BinStop=10;
    Data_MicroDop_2=0;
    RangeDop_2=0;

    for RBin=BinStart:1:BinStop %In this for we calculate the spectrogram for each range bin and sum the partial results

        Data_MicroDop_1 = fftshift(spectrogram(mean(Data_ForMicroDop(:,RBin),2),TimeWindowLength,OverlapLength,FFTPoints),1);
        Data_MicroDop_2=Data_MicroDop_2+abs(Data_MicroDop_1);

    end
    Data_MicroDop_2=flipud(Data_MicroDop_2);

    %                 figure;
    %                 imagesc(TimeAxisMD,DopplerAxisMD,20*log10(abs(Data_MicroDop_2)./max(abs(Data_MicroDop_2(:)))))
    %                 axis xy; set(gca,'FontSize',14);
    %                 colormap('jet');
    %                 title(file);
    %                 xlabel('Time [s]','FontSize',14); ylabel('Doppler [Hz]','FontSize',14);caxis([-40 0]);colorbar;
    %                 ylim([-20 20])   %% Cut of Spectrograms from -30 Hz to 30 HZ
    %
    %                 time_axis=TimeAxisMD;
    %
    %                 doppler_axis=20*log10(abs(Data_MicroDop_2)./max(abs(Data_MicroDop_2(:))));
    %
    %                 RangeAxis;




    %%

    % SAVE Spectrograms in a  Folder

    fpath = [outputfilefolder,'\'];
    % Data_MicroDop{:,:,NumofRep}=flipud(Data_MicroDop_2);  %index as cells
    % Data_MicroDop=Data_MicroDop(:,:,NumofRep);   %change to save arrays as a single variable

    save(strcat(fpath,'test_xethru'), 'Data_MicroDop_2','TimeAxisMD','DopplerAxisMD','Endtime','Starttime');

    % saveas(gca, fullfile(fpath, strcat('P0',num2str(NumofPeople),'SB',num2str(Numofclass),'R',num2str(NumofRep))), 'jpeg');

end