function gPars=getGpars(bnds,xys,dz)
%dx=xys(1);dy=xys(2);
x=bnds(1)+xys(1)/2:xys(1):bnds(2)-xys(1)/2;
y=bnds(3)+xys(2)/2:xys(2):bnds(4)-xys(2)/2;
if length(dz)==1
    Nz=floor((bnds(6)-bnds(5))/dz);
    dz=repmat(dz,1,Nz);
else
    Nz=length(dz);
end
z=zeros(1,Nz);
z(1)=bnds(5)+dz(1)/2;
for izc=2:Nz
    z(izc)=z(izc-1)+(dz(izc-1)+dz(izc))/2;
end
indOut=z>bnds(end);
z(indOut)=[];dz(indOut)=[];
[xg,yg,zg]=ndgrid(x,y,z);
[~,~,dzg]=ndgrid(x,y,dz);
gPars.x=x;gPars.y=y;gPars.z=z;gPars.dz=dz;
gPars.dx=xys(1);gPars.dy=xys(2);
gPars.xg=xg(:);gPars.yg=yg(:);gPars.zg=zg(:);gPars.dzg=dzg(:);
%gPars.Nx=length(x);gPars.Ny=length(y);gPars.Nz=length(z);
