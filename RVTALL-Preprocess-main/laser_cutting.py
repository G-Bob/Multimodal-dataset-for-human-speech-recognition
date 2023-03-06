import matplotlib.pyplot as plt
import matplotlib
import traceback

import os
import io

import sys
import csv

import glob
import tqdm
import json

import subprocess

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt

from scipy.io import loadmat
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from basic_proc import BasicProc
from bvh_converter.bvhplayer_skeleton import process_bvhfile, process_bvhkeyframe

ffmpeg_exe_path = r'C:\ffmpeg\bin\ffmpeg.exe'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.9f' % x)

class BVHReader:
    """
    BVHReader is used to extract 3D position data from mocap .bvh file.
    --------
    Args:
    --------
    save_folder: the target folder for save output csv files.
    ----------------
    Methods:
    --------
    open_csv: help to read csv file.
    proc_one_file: process a single bvh file.
    proc_folder: process all bvh files in a folder.
    """
    def __init__(self, save_folder):
        self.save_folder = save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  

    def open_csv(self, filename, mode='r'):
        """Open a csv file in proper mode depending on Python version."""
        if sys.version_info < (3,):
            return io.open(filename, mode=mode+'b')
        else:
            return io.open(filename, mode=mode, newline='')

    def proc_one_file(self, file_in, need_rot=False):

        if not os.path.exists(file_in):
            print("Error: file {} not found.".format(file_in))
            sys.exit(0)

        other_s = process_bvhfile(file_in)

        for i in range(other_s.frames):
            new_frame = process_bvhkeyframe(other_s.keyframes[i], other_s.root,
                                            other_s.dt * i)
        
        file_out = file_in[:-4] + ".csv"
        file_rot_out = ''

        with self.open_csv(self.save_folder+'/'+file_out.split('\\')[-1], 'w') as f:
            writer = csv.writer(f)
            header, frames = other_s.get_frames_worldpos()
            writer.writerow(header)
            for frame in frames:
                writer.writerow(frame)   

        if need_rot:
            file_rot_out = file_in[:-4] + "_rotations.csv"
        
            with self.open_csv(self.save_folder+'/'+file_rot_out.split('\\')[-1], 'w') as f:
                writer = csv.writer(f)
                header, frames = other_s.get_frames_rotations()
                writer.writerow(header)
                for frame in frames:
                    writer.writerow(frame)

    def proc_folder(self, tgt_folder, need_rot=False):
        err_file =[]
        for file_in in tqdm.tqdm(glob.glob(tgt_folder+'/*.bvh')):
            try:
                self.proc_one_file(file_in=file_in, need_rot=need_rot)
            except:
                print(r'error when working on: '+tgt_folder)
                err_file.append(file_in)
        

class KinectProcessor:
    """
    FacePt is used to process kinect-captured face points, especially, lip points.
    --------
    Args:
    --------
    root_dir: str, the directory saved kinect data.
    save_dir: str, the directory used to save extracted frame files.
    """
    def __init__(self, root_dir, save_dir):
        self.lip_pts_name = [
            'FaceJoint16',
            'FaceJoint27End', 'FaceJoint26End', 'FaceJoint25End', 
            'FaceJoint24End', 'FaceJoint23End', 'FaceJoint22End', 
            'FaceJoint21End','FaceJoint20End', 'FaceJoint19End', 
            'FaceJoint18End', 'FaceJoint17End','FaceJoint16End'
            ]
        
        self.face_pts_name = ['FaceJoint28'] + ['FaceJoint{}End'.format(id) for id in range(0, 29)]

        self.root_dir = root_dir
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)    

        if not os.path.exists(save_dir + '/videos'):
            os.makedirs(save_dir + '/videos')
            
        if not os.path.exists(save_dir + '/audios'):
            os.makedirs(save_dir + '/audios')

        if not os.path.exists(save_dir + '/movies'):
            os.makedirs(save_dir + '/movies')
            
        if not os.path.exists(save_dir + '/landmarkers'):
            os.makedirs(save_dir + '/landmarkers')

    def extract_frames_folder(self):
        """
        Given the root directory, extract frames of all files according to the start and end time
        and save extracted frames into a save_folder.
        """

        for file in tqdm.tqdm(glob.glob(self.root_dir+'/landmarkers/*.csv')):
            df = pd.read_csv(file)

            temp = file.split('\\')[-1].replace('.csv', '.json')
            temp = temp.replace('land', 'timestamp')

            timestamp_file = self.root_dir + '/timestamps/' + temp

            # extract landmakers csv file
            df_temp = self._extract_frames_onefile(df, timestamp_file)
            df_temp.to_csv(self.save_dir+'/landmarkers/'+file.split('\\')[-1].replace('land', 'land_proc'))

            # extract audio wav file
            temp = file.split('\\')[-1].replace('.csv', '.wav')
            temp = temp.replace('land', 'audio')
            end_time = (df_temp.Time[len(df_temp)-1] - df_temp.Time[0]) * 1000 # works in milliseconds
            new_audio = AudioSegment.from_wav(self.root_dir + '/audios/' + temp)
            new_audio = new_audio[:end_time]
            new_audio.export(self.save_dir+'/audios/'+temp.replace('audio', 'audio_proc'), format='wav')
            audio_name = self.save_dir+'/audios/'+temp.replace('audio', 'audio_proc')
            print(audio_name)

            # extract video avi file
            temp = file.split('\\')[-1].replace('.csv', '.avi')
            temp = temp.replace('land', 'video')

            # loading video 
            ffmpeg_extract_subclip(self.root_dir+'/videos/'+temp, 0, df_temp.Time[len(df_temp)-1] - df_temp.Time[0], 
                          targetname=self.save_dir+'/videos/'+temp.replace('video', 'video_proc'))
            video_name = self.save_dir+'/videos/'+temp.replace('video', 'video_proc')
            print(video_name)

            # combine video and audio
            command = ffmpeg_exe_path + ' -i ' + video_name + ' -i ' + audio_name + ' -c:v copy -c:a aac ' + self.save_dir + '/movies/' + temp.replace('video', 'movie')
            
            subprocess.run(command.replace('\\', '/'))


    def _extract_frames_onefile(self, df, timestamp_file):
        """
        Given the start time and end time, extract frames within the duration.
        --------
        Args:
        --------
        df: pandas.DataFrame, contains time step and 3d position information. The key of time step column should be 'Time'.
        timestamp_file: str, json file store start and end timestamps.
        """
        
        with open(timestamp_file, 'r') as f:
            times = json.load(f)

        timestamps = self._calcu_ts_onefile(df, times)

        for i, ts in enumerate(timestamps):
            if ts > times['end_time']:
                break

        final_df = df.iloc[0:i].copy()
        final_df.Time = timestamps[0:i]

        return final_df

    def _calcu_ts_onefile(self, df, times):
        """
        Given the start time, calculate the timestamps for every frames.
        --------
        Args:
        --------
        df: pandas.DataFrame, contains time step and 3d position information. The key of time step column should be 'Time'.
        times: dict, have keys: start_time, end_time, start_dtime, end_dtime.
        """

        timestamps = []

        start_time = times['start_time']

        for i in range(len(df)):

            timestamps.append(start_time+df.Time[i])

        return timestamps

    def ani_motion_onefile(self, file, mode='lip', pause=0.01):
        """
        Given a mocap data, animate movements of recorded points in 3d axes.
        --------
        Args:
        --------
        df: pandas.DataFrame, contains time step and 3d position information. The key of 3d position columns should be one of lip_pts_name/face_pts_name + '.X/Y/Z'.
        mode: str, availale options are: face and lip.
        pause: float, control animation play speed.
        """
        
        df = pd.read_csv(file)

        show_pts = []
        
        if mode == 'lip':
            show_pts = self.lip_pts_name.copy()
        elif mode == 'face':
            show_pts = self.face_pts_name.copy()
        else:
            raise Exception('Wrong mode: available modes 1. face, 2. lip')

        plt.ion() # enable interactive control
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for f in range(len(df)):
            ax.cla()

            x = [df[pt+'.X'][f] for pt in show_pts]
            y = [df[pt+'.Y'][f] for pt in show_pts]
            z = [df[pt+'.Z'][f] for pt in show_pts]
            ax.scatter(x, y, z)
            
            plt.draw()
            plt.pause(pause)

class UWBProcessor(BasicProc):
    """
    UWBProcessor is used to segment and preprocess UWB spectrogram
    --------
    Args:
    --------
    root_dir: str, the directory saved UWB data.
    """
    def __init__(self, root_dir):
        
        super(UWBProcessor, self).__init__()
        
        self.lookup_dict = {'1': 'vowel', '2': 'word', '3': 'sentences'}

        self.root_dir = root_dir.replace('\\', '/')
        
    def _segment_one_exp(self, uwbmat_file):
        """
        Parameters
        ----------
        uwbmat_file : str
            file name of UWB mat: EXPID_TASKID_PERSONID_RADARID_xethru.mat.

        Returns
        -------
        None.
        """
        exp_info = uwbmat_file.split('_')
        
        uwbmat = loadmat(self.root_dir + '/UWB_Person_' + exp_info[2] + '/' + uwbmat_file)
        timestamp_folder = self.root_dir + '/Kinect_Person_' + exp_info[2] + '/' + self.lookup_dict[exp_info[1]] + exp_info[0] + '/timestamps'
        save_dir = self.root_dir + '/UWB_Person_' + exp_info[2] + '/' + self.lookup_dict[exp_info[1]] + exp_info[0]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
        
        cnt = 1
        for timestamp_file in tqdm.tqdm(glob.glob(timestamp_folder + '/*.json')):
            try:
                uwb_sample = self._segment_one_sample(uwbmat, timestamp_file)
                np.save(save_dir + '/sample' + str(cnt) + '.npy', uwb_sample)
                cnt += 1
            except:
                cnt += 1
                continue
            
        
    def _segment_one_sample(self, uwbmat, timestamp_file):
        """
        Parameters
        ----------
        matfile : str
            location of UWB mat file.
        timestamp_file : str
            location of timestamp file.

        Returns
        -------
        uwb_sample : np.array
            a uwb spectrogram segment.
        """
        times = self._loadtimestamp(timestamp_file)
        frames_time = self._calcu_timestamp_uwbframes(uwbmat)
        
        kinec_start_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['start_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        kinec_end_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['end_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        
        start_idx, end_idx = self._match_start_end_ts(frames_time, kinec_start_time, kinec_end_time)
        
        uwb_sample = uwbmat['Data_MicroDop_2'][:, start_idx:end_idx+1]
        
        return uwb_sample
        
    def _calcu_timestamp_uwbframes(self, uwbmatobj):
        """
        Parameters
        ----------
        uwbmatobj : mat object
            contains UWB spectrogram and time information.

        Returns
        -------
        frames_time : list
            timestamp for each frame of UWB mat.
        """
        frames_time = [] 
        
        frames_no = uwbmatobj['Data_MicroDop_2'].shape[1]
        start_dtime_str = uwbmatobj['Starttime'][0]
        end_dtime_str = uwbmatobj['Endtime'][0]
        
        start_time = self._datetime2unixtimestamp(self._iosstr2datetime(start_dtime_str))
        end_time = self._datetime2unixtimestamp(self._iosstr2datetime(end_dtime_str))
        time_interval = end_time - start_time
        
        for i in range(frames_no):

            frames_time.append(start_time+i*time_interval/(frames_no-1))

        return frames_time
    
class mmWaveProcessor(BasicProc):
    
    def __init__(self, root_dir):
        super(mmWaveProcessor, self).__init__()
        
        self.lookup_dict = {'1': 'vowel', '2': 'word', '3': 'sentences'}

        self.root_dir = root_dir.replace('\\', '/')
     
        
    def _segment_one_exp(self, mmwmat_file):
        """
        Parameters
        ----------
        uwbmat_file : str
            file name of UWB mat: EXPID_TASKID_PERSONID_RADARID_xethru.mat.

        Returns
        -------
        None.
        """
        
        exp_info = mmwmat_file.replace('.mat', '').split('_')
        
        mmwmat = loadmat(self.root_dir + '/Preprocessing data' +'/mwradar/'+ '/output/'+ mmwmat_file)
        timestamp_root = self.root_dir +'/raw data/'+exp_info[0]+ '/'+exp_info[0]+'_kinect_uwb' 
        # timestamp_folder = self.root_dir + '/kinect/' + exp_info[0] +'_Kinect' + '/' + self.lookup_dict[exp_info[1]] + exp_info[2] + '/timestamps'
        save_dir = self.root_dir + '/Processed&cut data'+'/radar_processed/' + exp_info[0] + '/' + self.lookup_dict[exp_info[1]] + exp_info[2]
        
        sucess=[]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
        
        
        
        for i in os.listdir(timestamp_root):
            cnt = 1
            timestamp_folder = timestamp_root+ '/' + i + '/timestamps'
            print(timestamp_folder)
            for timestamp_file in tqdm.tqdm(glob.glob(timestamp_folder + '/*.json')):
                try:
                    
                    mmw_sample = self._segment_one_sample(mmwmat, timestamp_file)
                    sucess.append(mmwmat_file)
                    with open('sucess.csv','w')as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(sucess)
                    np.save(save_dir + '/sample' + str(cnt) + '.npy', mmw_sample)
                    cnt += 1
                except:
                    cnt += 1
                    continue
        
            
    def _segment_one_sample(self, mmwmat, timestamp_file):
        """
        Parameters
        ----------
        matfile : str
            location of UWB mat file.
        timestamp_file : str
            location of timestamp file.

        Returns
        -------
        uwb_sample : np.array
            a uwb spectrogram segment.
        """
        times = self._loadtimestamp(timestamp_file)
        frames_time = self._calcu_timestamp_mmWaveframes(mmwmat, timestamp_file)
        
        kinec_start_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['start_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        kinec_end_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['end_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        
        start_idx, end_idx = self._match_start_end_ts(frames_time, kinec_start_time, kinec_end_time)
        
        mmw_sample = np.abs(mmwmat['s'][:, start_idx:end_idx+1])
        
        return mmw_sample
        
    def _calcu_timestamp_mmWaveframes(self, mmwmatobj, timestamp_file):
        """
        Parameters
        ----------
        uwbmatobj : mat object
            contains mmWave spectrogram and time information.

        Returns
        -------
        frames_time : list
            timestamp for each frame of UWB mat.
        """
        times = self._loadtimestamp(timestamp_file)
        
        frames_time = [] 
        
        frames_no = mmwmatobj['s'].shape[1]
        start_dtime_str = times['start_dtime'][0:11] + mmwmatobj['TiTime'][0][0][0][0][0][1:13]
        end_dtime_str = times['start_dtime'][0:11] + mmwmatobj['TiTime'][0][0][2][0][0][1:13]
        
        start_time = self._datetime2unixtimestamp(self._iosstr2datetime(start_dtime_str, form="%Y-%m-%d %H:%M:%S:%f"))
        end_time = self._datetime2unixtimestamp(self._iosstr2datetime(end_dtime_str, form="%Y-%m-%d %H:%M:%S:%f"))
        time_interval = end_time - start_time
        
        for i in range(frames_no):

            frames_time.append(start_time+i*time_interval/(frames_no-1))

        return frames_time
    
        
class LaserProcessor(BasicProc):
    
    def __init__(self, root_dir):
        super(LaserProcessor, self).__init__()
        
        self.root_dir = root_dir.replace('\\', '/')
        
    def lookup_dict(self, index):
        """
        Parameters
        ----------
        index : int
            the index of task
        Returns
        -------
        task name, real index
        """
        if index >= 1 and index <=5:
            return 'vowel', index
        elif index >= 6 and index <= 20:
            return 'word', int(index-5)
        elif index >= 21 and index <= 30:
            return 'sentences', int(index-20)
        
    def _segment_one_exp(self, lasermat_file, index):
        """
        Parameters
        ----------
        uwbmat_file : str
            file name of UWB mat: EXPID_TASKID_PERSONID_RADARID_xethru.mat.
        Returns
        -------
        None.
        
        """

        exp_info = lasermat_file.replace('.mat', '').split('_')
        
        lasermat = loadmat(self.root_dir + r'/Preprocessing data/laser_data/' + lasermat_file)
        timestamp_folder = (r'D:\gy\mouth data\Raw data\1\1_kinect_uwb\vowel1\timestamps')
        # self.root_dir + r'/Raw data/' + str(int(exp_info[-1])) + '/'+ str(int(exp_info[-1])) + '_kinect_uwb' + '/' + self.lookup_dict(index)[0] + str(self.lookup_dict(index)[1]) + '/timestamps'
        save_dir = self.root_dir + r'/Processed&cut data/laser_processed/' + self.lookup_dict(index)[0] + str(self.lookup_dict(index)[1])
        

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
        
        cnt = 1
        for timestamp_file in tqdm.tqdm(glob.glob(timestamp_folder + '/*.json')):
            try:
                laser_sample = self._segment_one_sample(lasermat, timestamp_file, index)
                np.save(save_dir + '/', + exp_info[0]+'_'+ exp_info[1]+'_'+exp_info[2]+'_sample' + str(cnt) + '.npy', laser_sample)
                cnt += 1
            except:
                cnt += 1
                continue
            
    def _segment_one_sample(self, lasermatobj, timestamp_file, index):
        """
        Parameters
        ----------
        matfile : str
            location of UWB mat file.
        timestamp_file : str
            location of timestamp file.
        Returns
        -------
        uwb_sample : np.array
            a uwb spectrogram segment.
        """
        times = self._loadtimestamp(timestamp_file)
        frames_time = self._calcu_timestamp_laserframes(lasermatobj, index)
        
        kinec_start_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['start_dtime'], "%Y-%m-%d %H:%M:%S.%f"))
        kinec_end_time = self._datetime2unixtimestamp(self._iosstr2datetime(times['end_dtime'], "%Y-%m-%d %H:%M:%S.%f"))

        
        start_idx, end_idx = self._match_start_end_ts(frames_time, kinec_start_time, kinec_end_time)
        
        laser_sample = lasermatobj['lip_dataset_Y'][int(index-1)][start_idx:end_idx+1]
        
        return laser_sample
        
    def _calcu_timestamp_laserframes(self, lasermatobj, index):
        """
        Parameters
        ----------
        uwbmatobj : mat object
            contains mmWave spectrogram and time information.
        Returns
        -------
        frames_time : list
            timestamp for each frame of UWB mat.
        """
        frames_time = [] 
        
        frames_no = lasermatobj['lip_dataset_Y'].shape[1]
        # start_dtime_str = '-'.join([str(int(x)) for x in lasermatobj['datatimestart'][int(index-1)][0:-1]] + [str(lasermatobj['datatimestart'][int(index-1)][-1])])
        end_dtime_str = '-'.join([str(int(x)) for x in lasermatobj['datatimestop'][int(index-1)][0:-1]] + [str(lasermatobj['datatimestop'][int(index-1)][-1])])
        
        # start_time = self._datetime2unixtimestamp(self._iosstr2datetime(start_dtime_str, form="%Y-%m-%d-%H-%M-%S.%f"))
        end_time = self._datetime2unixtimestamp(self._iosstr2datetime(end_dtime_str, form="%Y-%m-%d-%H-%M-%S.%f"))
        time_interval = lasermatobj['datatimestep'][int(index-1)][-1]
        

        for i in reversed(range(frames_no)):

            frames_time.append(end_time-lasermatobj['datatimestep'][int(index-1)][i])

        return frames_time

if __name__ == '__main__':
    root = r'D:\gy\mouth data'
    radar_root = r'D:\gy\mouth data\Preprocessing data\laser_data'
    
    timestampfileslist=[]
    
    laser_proc = LaserProcessor(root_dir=root)
    err=[]
    
    #lasermatobj = loadmat(r'D:\gy\mouth data\Preprocessing data\laser_data\Laser_data_person_01.mat')
    laser_proc._segment_one_exp('Laser_data_person_01.mat', 1)
 
    '''
    radarfiles = os.listdir(radar_root)
    
    for f in radarfiles:            
        # exp_info = f.replace('.mat', '').split('_')

        #try:
        for idx in range(1, 31):                    
            laser_proc._segment_one_exp(f, idx)                
                
       # except:
    '''   #     err.append(f)

