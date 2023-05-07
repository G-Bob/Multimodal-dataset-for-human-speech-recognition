function [time_stamp] = json_read(addr)
% read the contents of the JSON file
json_str = fileread(addr);

% decode the JSON data
tsdata = jsondecode(json_str);
time_stamp = tsdata(1);
end

