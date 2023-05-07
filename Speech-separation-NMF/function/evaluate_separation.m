function [SDR, SIR] = evaluate_separation(s_true, s_est, eps)
% Computes the SDR and SIR between two signals
% Inputs:
%   s_true: true source signal
%   s_est: estimated source signal
%   eps: small positive constant to avoid division by zero
% Outputs:
%   SDR: signal-to-distortion ratio
%   SIR: signal-to-interference ratio

% Compute the energy of the true source signal
E_true = norm(s_true)^2;

% Compute the energy of the error signal
e = s_true - s_est;
E_error = norm(e)^2;

% Compute the energy of the interference signal
i = s_est - mean(s_est);
E_interf = norm(i)^2;

% Compute the SDR and SIR
SDR = 10*log10(E_true/(E_error+eps));
SIR = 10*log10(E_true/(E_interf+eps));
disp('***********************************')
end
