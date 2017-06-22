function [ R ] = RotationEstimate( Xo, Xg )
%ROTATIONESTIMATE Summary of this function goes here
%   Detailed explanation goes here
Xom = mean(Xo')';
Xgm = mean(Xg')';
Xos = Xo - repmat(Xom,1,size(Xo,2));
Xgs = Xg - repmat(Xgm,1,size(Xg,2));
H = Xos*Xgs';
[U,D,V] = svd(H);
X = V*U';
dX = det(X);
if (dX < 0)
    R =  zeros(3,3);
else
    R = X;
end


end

