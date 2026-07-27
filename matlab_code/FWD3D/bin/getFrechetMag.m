function [Frechet,predData]=getFrechetMag(xq,yq,zq,w,m,rx,ry,rz,rc,Bo,Ao,Io,Do)
% Y-N, X-E, Z-DOWN (LHS)
lx=cos(pi*Io/180)*sin(pi*(Do-Ao)/180);
ly=cos(pi*Io/180)*cos(pi*(Do-Ao)/180);
lz=sin(pi*Io/180);
gamma=Bo.*w/(4*pi);
Nc=size(xq,1);
Nr=length(rx);
Ncm=length(rc);
Frechet=zeros(Nr*Ncm,Nc);
for ir=1:Nr
    xx=xq-rx(ir);
    yy=yq-ry(ir);
    zz=zq-rz(ir);
    R2=xx.^2+yy.^2+zz.^2;
    R=sqrt(R2);
    lxd=lx*xx+ly*yy+lz*zz;
    for icm=1:Ncm
        indc=icm+Ncm*(ir-1);
        switch rc(icm)
            case 1%TMI
                Frechet(indc,:)=sum(gamma.*(3*lxd.^2./R2-1)./(R2.*R),2);
            case 2%Hx
                Frechet(indc,:)=sum(gamma.*(3*lxd.*xx./R2-lx)./(R2.*R),2);
            case 3%Hy
                Frechet(indc,:)=sum(gamma.*(3*lxd.*yy./R2-ly)./(R2.*R),2);
            case 4%Hz
                Frechet(indc,:)=sum(gamma.*(3*lxd.*zz./R2-lz)./(R2.*R),2);
        end
    end
end
predData=Frechet*m;
