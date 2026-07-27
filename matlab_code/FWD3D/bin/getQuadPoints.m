function [xq,yq,zq,w]=getQuadPoints(xc,yc,zc,dx,dy,dz,flag)
switch flag
    case 1%pulse basis
        xq=xc;
        yq=yc;
        zq=zc;
        w=dx.*dy.*dz;
    case 2%Gaussian points on rectangular prisms
    case 3%Gaussian points on triangular prisms
end