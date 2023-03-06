import json

import datetime

class BasicProc:
    def __init__(self):
        pass
    
    def _match_start_end_ts(self, frames_time, start_time, end_time):
        """
        Parameters
        ----------
        frames_time : List
            save timestamps for each spectrogram frame.
        start_time : float
            start unix time.
        end_time : float
            end unix time.

        Returns
        -------
        start_idx : int
            segmented spectrogram start index.
        end_idx : int
            segmented spectrogram end index.
        """
        if start_time > frames_time[-1] or start_time < frames_time[0] or end_time > frames_time[-1] or end_time < frames_time[0]:
            raise Exception("Start/End time out of bound.")
            
        temp_start_diff = [abs(t-start_time) for t in frames_time]
        temp_end_diff = [abs(t-end_time) for t in frames_time]
        
        start_idx = temp_start_diff.index(min(temp_start_diff))
        end_idx = temp_end_diff.index(min(temp_end_diff))
        
        return start_idx, end_idx
    
    def _loadtimestamp(self, timestamp_file):
        """
        Parameters
        ----------
        timestamp_file : str
            location of timestamp file.

        Returns
        -------
        times : dict
            save timestamp information.
        """
        with open(timestamp_file, 'r') as f:
            times = json.load(f)
            
        return times
    
    def _iosstr2datetime(self, dt_str, form="%d-%m-%Y %H:%M:%S %f"):
        """
        Parameters
        ----------
        dt_str : str
            isoformat string "%d-%m-%Y %H:%M:%S %f".

        Returns
        -------
        datetime: datetime.datetime
        """
        return datetime.datetime.strptime(dt_str, form)
    
    def _datetime2unixtimestamp(self, dt):
        """
        Parameters
        ----------
        dt : datetime.datetime
            date time.

        Returns
        -------
        timestamp : float
            unix timestamp.
        """
        return datetime.datetime.timestamp(dt)