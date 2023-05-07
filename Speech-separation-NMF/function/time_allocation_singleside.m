function time = time_allocation_singleside(seq1,seq2,seq3)
% Check if the two sequences overlap
    time = min([seq1(end), seq2(end), seq3(end)]);
    fprintf('The two sequences overlap from %d to %d.\n', time);
end