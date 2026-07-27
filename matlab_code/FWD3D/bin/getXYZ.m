function [x,y,z,dz]=getXYZ(bnds,xys,dz)
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