function [cropped_seq_index] = crop_mintime_sequence(seq1_t, seq2_t, seq3_t, min_time)
% Crop the two time sequences based on the crop timestamps

% Crop the first time sequence
if seq1_t(end) == min_time 
    cropped_seq1_index = find(seq1_t == seq1_t(end));
    cropped_seq2_index = find(seq2_t > min_time,1,'first') - 1;
    cropped_seq3_index = find(seq3_t > min_time,1,'first') - 1; 
elseif seq2_t(end) == min_time 
    cropped_seq1_index = find(seq1_t > min_time,1,'first') - 1;
    cropped_seq2_index = find(seq2_t == seq2_t(end));
    cropped_seq3_index = find(seq3_t > min_time,1,'first') - 1; 
elseif seq3_t(end) == min_time 
    cropped_seq1_index = find(seq1_t > min_time,1,'first') - 1;
    cropped_seq2_index = find(seq2_t > min_time,1,'first') - 1;
    cropped_seq3_index = find(seq3_t == seq3_t(end));
else
    error('sth wrong with minimum time')
end

cropped_seq_index = min([cropped_seq1_index,cropped_seq3_index,cropped_seq2_index]);

% % Check if the cropped sequences are within the original time range
% if cropped_seq1_start_time < seq1_start_time || cropped_seq1_stop_time > seq1_stop_time || cropped_seq2_start_time < seq2_start_time || cropped_seq2_stop_time > seq2_stop_time
%     error('The cropped time sequences are outside of the original time range.');
% end

end
