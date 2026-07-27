function predData=getPredGR(xq,yq,zq,w,m,rx,ry,rz,rc)
gamma=1e8*6.67e-11.*w;
Nr=length(rx);
Ncm=length(rc);
predData=zeros(Nr*Ncm,1);
for ir=1:Nr
    xx=xq-rx(ir);
    yy=yq-ry(ir);
    zz=zq-rz(ir);
    R2=xx.^2+yy.^2+zz.^2;
    R=sqrt(R2);
    for ic=1:Ncm
        indc=ic+Ncm*(ir-1);
        switch rc(ic)
            case 1%Gt
                predData(indc)=dot(sum(gamma./R2,2),m);
            case 2%Gx
                predData(indc)=dot(sum(gamma.*xx./(R.*R2),2),m);
            case 3%Gy
                predData(indc)=dot(sum(gamma.*yy./(R.*R2),2),m);
            case 4%Gz
                predData(indc)=dot(sum(gamma.*zz./(R.*R2),2),m);
            case 5%Gxx
                predData(indc)=dot(1e4.*sum(gamma.*(2*xx.^2-yy.^2-zz.^2)./(R.*R2.*R2),2),m);
            case 6%Gyy
                predData(indc)=dot(1e4.*sum(gamma.*(2*yy.^2-xx.^2-zz.^2)./(R.*R2.*R2),2),m);
            case 7%Gzz
                predData(indc)=dot(1e4.*sum(gamma.*(2*zz.^2-xx.^2-yy.^2)./(R.*R2.*R2),2),m);
            case 8%Gxy
                predData(indc)=dot(3e4.*sum(gamma.*xx.*yy./(R.*R2.*R2),2),m);
            case 9%Gzx
                predData(indc)=dot(3e4.*sum(gamma.*xx.*zz./(R.*R2.*R2),2),m);
            case 10%Gzy
                predData(indc)=dot(3e4.*sum(gamma.*zz.*yy./(R.*R2.*R2),2),m);
            case 11%Gd=0.5*(Gxx-Gyy)
                predData(indc)=dot(1.5e4.*sum(gamma.*(xx.^2-yy.^2)./(R.*R2.*R2),2),m);
        end
    end
end
