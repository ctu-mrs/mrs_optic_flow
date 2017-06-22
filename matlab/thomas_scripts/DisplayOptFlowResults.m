%function [ output_args ] = DisplayOptFlowResults( FileName, FigNum )
%DISPLAYOPTFLOWRESULTS Summary of this function goes here
%   Detailed explanation goes here

close all;
clear;

%% SET variables
folder= '~/ros_workspace/src/MRS-OpticFlow/optic_flow/bags/eval_out/';

FileName = strcat(folder,'SpeedCompareM4B80R4S0F240-noFB_ALLSAC_FFT_CBM3_NOYAW_9WIN_RAD1_wSTDDEV.bag');
Name = 'FFT noFB CBM3 9 WINDOW NOYAW ALLSAC-RAD 1m/s';
figName = 'noFB_fft_cbm3_9win_noYaw_allsac_rad_1m-s';% '_fft_cbm0_1win_yaw';

fromTime = 0;%63.69; % cbm3 no yaw
%fromTime = 53.3; % cbm3 yaw
%fromTime = 30.75;% cbm0 yaw
%fromTime = 0;
FigNum = 2;

stdDEV = true;

%% LOAD rosbags

disp('Loading data...');

RB = ros.Bag.load(FileName);

[msgs_odom,meta_odom] = RB.readAll('/uav5/mbzirc_odom/new_odom');
[msgs_flow,meta_flow] = RB.readAll('/optFlow/velocity');
[msgs_flraw,meta_flraw] = RB.readAll('/optFlow/velocity_raw');
if(stdDEV)
    [msgs_sddev,meta_sddev] = RB.readAll('/optFlow/velocity_stddev');
end

% load flow node
flow_time = ros.msgs2mat(meta_flow, @(time) time.time.time); % time
start_time = flow_time(1);
flow_time = flow_time - start_time;

timeMask = flow_time >= fromTime; % create time mask
flow_time = flow_time(timeMask);

flow_linear = ros.msgs2mat(msgs_flow, @(linear) linear.linear); % optic flow
flow_linear = flow_linear(:,timeMask); % crop time


flow_height = ros.msgs2mat(msgs_flow, @(angular) angular.angular); % corrected altitude
flow_height = flow_height(3,:);

flow_height = flow_height(:,timeMask); % crop time

% load flowraw
flowraw_time = ros.msgs2mat(meta_flraw, @(time) time.time.time); % time
start_time = flowraw_time(1);
flowraw_time = flowraw_time - start_time;

timeMask = flowraw_time >= fromTime; % create time mask

flowraw_time = flowraw_time(timeMask);

flowraw_linear = ros.msgs2mat(msgs_flraw, @(linear) linear.linear);
flowraw_linear = flowraw_linear(:,timeMask);

% load odometry
odom_time =   ros.msgs2mat(meta_odom, @(time) time.time.time); % odom
odom_time = odom_time - start_time;

timeMask = odom_time >= fromTime; % time mask
odom_time = odom_time(timeMask); % crop time

odom_rot = ros.msgs2mat(msgs_odom, @(pose) pose.pose.pose.orientation);
odom_linear = ros.msgs2mat(msgs_odom, @(twist) twist.twist.twist.linear);
odom_angular = ros.msgs2mat(msgs_odom, @(twist) twist.twist.twist.angular);
odom_position = ros.msgs2mat(msgs_odom, @(pose) pose.pose.pose.position);
odom_height = odom_position(3,:);

odom_height = odom_height(:,timeMask); % crop odom
odom_rot = odom_rot(:,timeMask);
odom_linear = odom_linear(:,timeMask);
odom_position = odom_position(:,timeMask);
odom_yaw_vel = odom_angular(3,timeMask);

% load SD
if(stdDEV)
    sddev_time = ros.msgs2mat(meta_sddev, @(time) time.time.time); % time
    sddev_time = sddev_time - sddev_time(1);

    timeMask = sddev_time >= fromTime; % create time mask
    sddev_time = sddev_time(timeMask);

    sddev = ros.msgs2mat(msgs_sddev, @(linear) linear);
    sddev = sddev(:,timeMask);

    sddev_ts = timeseries(sddev,sddev_time);
    sddev_res = resample(sddev_ts,odom_time);
    sddev_odomTime = sddev_res.Data;
    sddev_odomTime = reshape(sddev_odomTime(1:2,:,:),[2,size(sddev_odomTime,3)]);

    sddev_max = odom_linear(1:2,:) + sddev_odomTime;
    sddev_min = odom_linear(1:2,:) - sddev_odomTime;
end
%% DO CORRECTIONS

% [roll, pitch, yaw] = quat2angle(odom_rot');
% yaw_ts = timeseries(yaw,odom_time);
% yaw_ts_res = resample(yaw_ts,flow_time);
% yaw = yaw_ts_res.Data;
% 
% yaw_vel_ts = timeseries(odom_yaw_vel,odom_time);
% yaw_vel_res = resample(yaw_vel_ts,flow_time);
% yaw_vel = yaw_vel_res.Data;
% 
% for i = (1:size(flow_linear,2))
%     
%     if isnan(yaw_vel(i))
%         if (i == 1)
%             yaw_vel(i) = 0;
%         else
%             yaw_vel(i) = yaw_vel(i-1);
%         end
%     end
%     
%    %(yaw_vel(i)*0.11)/flowraw_linear(1,i)
%     flowraw_linear(1,i) = flowraw_linear(1,i) + yaw_vel(i)*0.11;
%     
%     if isnan(yaw(i))
%         if (i == 1)
%             yaw(i) = pi();
%         else
%             yaw(i) = yaw(i-1);
%         end
%     end
%     phi = -(-yaw(i)+0.2+pi/2);
%    RotMatZ = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0 ; 0 0 1];
%    %flow_linear(1:3,i) = (RotMatZ*flow_linear(1:3,i));
%    flowraw_linear(1:3,i) = (RotMatZ*flowraw_linear(1:3,i));
% end

% R = R_corr;
% for i = (1:size(flow_linear,2))
%    flow_linear_corr(1:2,i) = (R*flow_linear(1:2,i));
% %    flowraw_linear_corr(1:2,i) = (R*flowraw_linear(1:2,i));
% end
% 
% % flow_linear(1:2,:) = flow_linear_corr(1:2,:);
% flowraw_linear(1:2,:) = flowraw_linear_corr(1:2,:);

%% PLOT VELOCITY

disp('Plotting velocity...');

figure(FigNum*10+1);

subplot(3,1,1);
if(stdDEV)
    fill([odom_time fliplr(odom_time)],[sddev_max(1,:) fliplr(sddev_min(1,:))],'b')
end

grid on
title(Name);
hold on;

plot(odom_time,odom_linear(1,:));

plot(flow_time,flow_linear(1,:));
%plot(flowraw_time,flowraw_linear(1,:));
if(stdDEV)
    legend('SdDev','Odom','Flow')%,'Flow raw')
else
    legend('Odom','Flow')
end
% plot(flow_time,flow_linear_corr(1,:));

subplot(3,1,2);

if(stdDEV)
    fill([odom_time fliplr(odom_time)],[sddev_max(2,:) fliplr(sddev_min(2,:))],'b')
end

hold on;

plot(odom_time,odom_linear(2,:));

plot(flow_time,flow_linear(2,:));
%plot(flowraw_time,flowraw_linear(2,:));

if(stdDEV)
    legend('SdDev','Odom','Flow')%,'Flow raw')
else
    legend('Odom','Flow')
end

% plot(flow_time,flow_linear_corr(2,:));
subplot(3,1,3);
plot(odom_time,odom_height);
hold;
plot(flow_time,flow_height);

savefig(FigNum*10+1,strcat('eval_figs/speed',figName));


%% INTEGRATE

disp('Integrating');


x_int = cumsum(flow_linear(1,:) .* diff([flow_time flow_time(end)]));
y_int = cumsum(flow_linear(2,:) .* diff([flow_time flow_time(end)]));
x_int_raw = cumsum(flowraw_linear(1,:) .* diff([flowraw_time flowraw_time(end)]));
y_int_raw = cumsum(flowraw_linear(2,:) .* diff([flowraw_time flowraw_time(end)]));

odom_position(1,:) = odom_position(1,:) - odom_position(1,1);
odom_position(2,:) = odom_position(2,:) - odom_position(2,1);
odom_position(3,:) = odom_position(3,:) - odom_position(3,1);

%% PLOT POSITION
disp('Plotting position...');

figure(FigNum*10+2);

subplot(2,1,1);



plot(odom_time,odom_position(1,:));
hold;
plot(flow_time,x_int);
%plot(flowraw_time,x_int_raw);
legend('Odom','Flow')%,'Flow raw')
title(Name);
subplot(2,1,2);
plot(odom_time,odom_position(2,:));
hold;
plot(flow_time,y_int);
%plot(flowraw_time,y_int_raw);
legend('Odom','Flow')%,'Flow raw')

savefig(FigNum*10+2,strcat('eval_figs/position',figName));


