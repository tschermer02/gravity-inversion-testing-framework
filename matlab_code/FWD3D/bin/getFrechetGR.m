function [Frechet,predData]=getFrechetGR(xq,yq,zq,w,m,rx,ry,rz,rc)
gamma=1e8*6.67e-11.*w;
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
    for ic=1:Ncm
        indc=ic+Ncm*(ir-1);
        switch rc(ic)
            case 1%Gt
                Frechet(indc,:)=sum(gamma./R2,2);
            case 2%Gx
                Frechet(indc,:)=sum(gamma.*xx./(R.*R2),2);
            case 3%Gy
                Frechet(indc,:)=sum(gamma.*yy./(R.*R2),2);
            case 4%Gz
                Frechet(indc,:)=sum(gamma.*zz./(R.*R2),2);
            case 5%Gxx
                Frechet(indc,:)=1e4.*sum(gamma.*(2*xx.^2-yy.^2-zz.^2)./(R.*R2.*R2),2);
            case 6%Gyy
                Frechet(indc,:)=1e4.*sum(gamma.*(2*yy.^2-xx.^2-zz.^2)./(R.*R2.*R2),2);
            case 7%Gzz
                Frechet(indc,:)=1e4.*sum(gamma.*(2*zz.^2-xx.^2-yy.^2)./(R.*R2.*R2),2);
            case 8%Gxy
                Frechet(indc,:)=3e4.*sum(gamma.*xx.*yy./(R.*R2.*R2),2);
            case 9%Gzx
                Frechet(indc,:)=3e4.*sum(gamma.*xx.*zz./(R.*R2.*R2),2);
            case 10%Gzy
                Frechet(indc,:)=3e4.*sum(gamma.*zz.*yy./(R.*R2.*R2),2);
            case 11%Gd=0.5*(Gxx-Gyy)
                Frechet(indc,:)=1.5e4.*sum(gamma.*(xx.^2-yy.^2)./(R.*R2.*R2),2);
        end
    end
end
predData=Frechet*m;
