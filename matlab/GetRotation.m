function [ R ] = GetRotation( FileName )
%GETROTATION Summary of this function goes here
%   Detailed explanation goes here



RB = ros.Bag.load(FileName);

[msgs_odom,meta_odom] = RB.readAll('/uav5/mbzirc_odom/new_odom');
[msgs_flow,meta_flow] = RB.readAll('/optFlow/velocity');

flow_time =   ros.msgs2mat(meta_flow, @(time) time.time.time);
start_time = flow_time(1);
flow_time = flow_time - start_time;

flow_linear = ros.msgs2mat(msgs_flow, @(linear) linear.linear);
flow_linear(1,:) = -flow_linear(1,:);

odom_time =   ros.msgs2mat(meta_odom, @(time) time.time.time);
odom_time = odom_time - start_time;

odom_rot = ros.msgs2mat(msgs_odom, @(pose) pose.pose.pose.orientation);
odom_linear = ros.msgs2mat(msgs_odom, @(twist) twist.twist.twist.linear);
odom_position = ros.msgs2mat(msgs_odom, @(pose) pose.pose.pose.position);

[yaw, pitch, roll] = quat2angle(odom_rot');
s = size(odom_rot,2);

for i = (1:s)
   RotMatZ = rotz(deg2rad(-yaw(i)));
   odom_linear(1:3,i) = (RotMatZ*odom_linear(1:3,i));
   odom_position(1:3,i) = (RotMatZ)*odom_position(1:3,i);
end

odom_ts = timeseries(odom_linear,odom_time);
odom_ts_res = resample(odom_ts,flow_time);
odom_res = odom_ts_res.Data;
startT = 0;
stopT = 1000;

 Io = find(flow_time > startT & flow_time < stopT);
%  plot(flow_time(Io),flow_linear(1,Io));
%  hold
%  plot(flow_time(Io),odom_res(1,Io));

R = RotationEstimate(flow_linear(1:2,Io),odom_res(1:2,Io));

end

