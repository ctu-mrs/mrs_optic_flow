function [ output_args ] = DisplayOptFlowResults( FileName, FigNum )
%DISPLAYOPTFLOWRESULTS Summary of this function goes here
%   Detailed explanation goes here
startOffset = 0;
duration = 250;


RB = ros.Bag.load(FileName);

[msgs_odom,meta_odom] = RB.readAll('/uav5/mbzirc_odom/new_odom');
[msgs_flow,meta_flow] = RB.readAll('/optFlow/velocity');
[msgs_flraw,meta_flraw] = RB.readAll('/optFlow/velocity_raw');


flow_time =   ros.msgs2mat(meta_flow, @(time) time.time.time);
start_time = flow_time(1);
flow_time = flow_time - start_time;

flow_linear = ros.msgs2mat(msgs_flow, @(linear) linear.linear);
flowraw_linear = ros.msgs2mat(msgs_flraw, @(linear) linear.linear);

flow_height = ros.msgs2mat(msgs_flow, @(angular) angular.angular);
flow_height = flow_height(3,:);

odom_time =   ros.msgs2mat(meta_odom, @(time) time.time.time);
odom_time = odom_time - start_time;

odom_rot = ros.msgs2mat(msgs_odom, @(pose) pose.pose.pose.orientation);
odom_linear = ros.msgs2mat(msgs_odom, @(twist) twist.twist.twist.linear);
odom_position = ros.msgs2mat(msgs_odom, @(pose) pose.pose.pose.position);
odom_height = odom_position(3,:);

[yaw, pitch, roll] = quat2angle(odom_rot');
yaw_ts = timeseries(yaw,odom_time);
yaw_ts_res = resample(yaw_ts,flow_time);
yaw = yaw_ts_res.Data;

for i = (1:size(flow_linear,2))
    if isnan(yaw(i))
        if (i == 1)
            yaw(i) = pi();
        else
            yaw(i) = yaw(i-1);
        end
    end
   RotMatZ = rotz(deg2rad(yaw(i)));
   flow_linear(1:3,i) = (RotMatZ*flow_linear(1:3,i));
   flowraw_linear(1:3,i) = (RotMatZ*flowraw_linear(1:3,i));
end

% R = R_corr;
% for i = (1:size(flow_linear,2))
%    flow_linear_corr(1:2,i) = (R*flow_linear(1:2,i));
% %    flowraw_linear_corr(1:2,i) = (R*flowraw_linear(1:2,i));
% end
% 
% % flow_linear(1:2,:) = flow_linear_corr(1:2,:);
% flowraw_linear(1:2,:) = flowraw_linear_corr(1:2,:);

figure(FigNum*10+1);
subplot(3,1,1);
grid on
plot(odom_time,odom_linear(1,:));
hold;
plot(flow_time,flow_linear(1,:));
plot(flow_time,flowraw_linear(1,:));
% plot(flow_time,flow_linear_corr(1,:));
subplot(3,1,2);
plot(odom_time,odom_linear(2,:));
hold;
plot(flow_time,flow_linear(2,:));
plot(flow_time,flowraw_linear(2,:));
% plot(flow_time,flow_linear_corr(2,:));
subplot(3,1,3);
plot(odom_time,odom_height);
hold;
plot(flow_time,flow_height);

figure(FigNum*10+2);
x_int = Integrate(flow_linear(1,:),flow_time);
y_int = Integrate(flow_linear(2,:),flow_time);
x_int_raw = Integrate(flowraw_linear(1,:),flow_time);
y_int_raw = Integrate(flowraw_linear(2,:),flow_time);
% x_int_corr = Integrate(flow_linear_corr(1,:),flow_time);
% y_int_corr = Integrate(flow_linear_corr(2,:),flow_time);
subplot(2,1,1);
plot(odom_time,odom_position(1,:));
hold;
plot(flow_time,x_int);
plot(flow_time,x_int_raw);
% plot(flow_time,x_int_corr);
subplot(2,1,2);
plot(odom_time,odom_position(2,:));
hold;
plot(flow_time,y_int);
plot(flow_time,y_int_raw);
% plot(flow_time,y_int_corr);


