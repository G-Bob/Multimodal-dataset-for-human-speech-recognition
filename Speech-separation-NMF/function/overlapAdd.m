function [y] = overlapAdd(frames, overlap)
% OVERLAPADD Reconstructs a signal from overlapping frames using the overlap-add method
%
%   y = OVERLAPADD(frames, overlap) reconstructs a signal from the given overlapping
%   frames, using the overlap-add method. The frames are assumed to be in columns of the
%   input matrix, with a fixed overlap between adjacent frames.
%
%   frames:  matrix of overlapping frames, with each column representing a frame
%   overlap: number of samples of overlap between adjacent frames
%
%   y:       reconstructed signal

% Get the frame length and number of frames
[frameLength, numFrames] = size(frames);

% Compute the shift amount between adjacent frames
shift = frameLength - overlap;

% Compute the length of the reconstructed signal
signalLength = frameLength + (numFrames - 1) * shift;

% Initialize the reconstructed signal to zeros
y = zeros(signalLength, 1);

% Initialize the index for the start of each frame
startIndex = 1;

% Iterate over each frame, adding it to the reconstructed signal
for i = 1:numFrames
    % Get the current frame
    frame = frames(:, i);
    
    % Add the frame to the reconstructed signal
    y(startIndex:startIndex+frameLength-1) = y(startIndex:startIndex+frameLength-1) + frame;
    
    % Update the index for the start of the next frame
    startIndex = startIndex + shift;
end
