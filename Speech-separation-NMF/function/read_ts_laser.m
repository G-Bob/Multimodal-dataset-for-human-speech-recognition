function [posix_time_ms] = read_ts_laser(addr_ts, index_iter, type)

ts_all = matfile(addr_ts);
if type == "starttime"
    ts = strjoin(ts_all.starttime(1,index_iter));
elseif type == "stoptime"
    ts = strjoin(ts_all.stoptime(1,index_iter));
end
ts_list = [ts(1:4),'-',ts(5:6),'-',ts(7:8),' ',ts(9:10),':',ts(11:12),':',ts(13:14),'.',ts(15:20)];
serial_date_num = datenum(ts_list);
posix_time_ms = posixtime(datetime(serial_date_num,'ConvertFrom','datenum','TimeZone','UTC'))*1000;

end

