function predData=getPredMag(xq,yq,zq,w,m,rx,ry,rz,rc,Bo,Ao,Io,Do)
% Y-N, X-E, Z-DOWN (LHS)
lx=cos(pi*Io/180)*sin(pi*(Do-Ao)/180);
ly=cos(pi*Io/180)*cos(pi*(Do-Ao)/180);
lz=sin(pi*Io/180);
gamma=Bo.*w/(4*pi);
Nr=length(rx);
Ncm=length(rc);
predData=zeros(Nr*Ncm,1);
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
                predData(indc)=dot(sum(gamma.*(3*lxd.^2./R2-1)./(R2.*R),2),m);
            case 2%Hx
                predData(indc)=dot(sum(gamma.*(3*lxd.*xx./R2-lx)./(R2.*R),2),m);
            case 3%Hy
                predData(indc)=dot(sum(gamma.*(3*lxd.*yy./R2-ly)./(R2.*R),2),m);
            case 4%Hz
                predData(indc)=dot(sum(gamma.*(3*lxd.*zz./R2-lz)./(R2.*R),2),m);
        end
    end
end
