clear all
close all
clc


% R =GetRotation('SpeedCompareB8R15S20F240.bag')
% alpha = asin(R(2,1))*(180/pi())
%R = [cos(alpha),-sin(alpha);sin(alpha),cos(alpha)];

% 
% DisplayOptFlowResults('SpeedCompareB240R4S0F240.bag',1,R);
% DisplayOptFlowResults('SpeedCompareB120R4S0F240.bag',2,R);
% DisplayOptFlowResults('SpeedCompareB80R4S0F240.bag',3,R);
% 
% DisplayOptFlowResults('SpeedCompareB8R15S20F240-.bag',2);
DisplayOptFlowResults('SpeedCompareB480R4S0F480-FFT-yaw-noBin.bag',1);
%DisplayOptFlowResults('SpeedCompareB8R8S0F240.bag',2);
%DisplayOptFlowResults('SpeedCompareB8R15S3F240.bag',3);
%DisplayOptFlowResults('SpeedCompareB8R4S0F64.bag',2);
%DisplayOptFlowResults('SpeedCompareB8R4S0F150.bag',3);
