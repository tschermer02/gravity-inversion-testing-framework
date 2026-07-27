function [Frechet,predData]=getFrechetAEM(sigb,thb,bnds,xys,dz,sigt,...
    rx,ry,rz,rc,iter,workdir)
[x,y,z,dz]=getXYZ(bnds,xys,dz);
freq=[56e3 7200 900 5500 900];
offset=[6.3 0 0; 8 0 0; 8 0 0; 8 0 0; 8 0 0]';
theta=[90 90 90 90 90];
phi=[90 90 90 0 0];
xshift=round((offset(1,:).*cos(theta*pi/180)-...
    offset(2,:).*sin(theta*pi/180))*100)/100;
yshift=round((offset(1,:).*sin(theta*pi/180)+...
    offset(2,:).*cos(theta*pi/180))*100)/100;
zshift=offset(3,:)';
lx=(cos(phi*pi/180)+1-1).*(cos(theta*pi/180)+1-1);
ly=(cos(phi*pi/180)+1-1).*(sin(theta*pi/180)+1-1);
lz=sin(phi*pi/180)+1-1;
freq=freq(rc);
xshift=xshift(rc);
yshift=yshift(rc);
zshift=zshift(rc);
lx=lx(rc);
ly=ly(rc);
lz=lz(rc);
Nc=length(x)*length(y)*length(z);
Nr=length(rx);
Nf=length(freq);
if ~iter
    predData=complex(zeros(Nr,Nf));
    Ho=zeros(Nr,Nf);
    for ifr=1:Nf
        trans=[rx-xshift(1)/2 ry-yshift(1)/2 rz-zshift(1)/2];
        Hx=complex(zeros(Nr,1));
        Hy=complex(zeros(Nr,1));
        Hz=complex(zeros(Nr,1));
        parfor ir=1:Nr
            par=[10 0 0 trans(ir,3) theta(ifr) phi(ifr)];
            [~,h]=green3d(freq(ifr),thb,sigb,ones(size(sigb)),...
                xshift(ifr),yshift(ifr),trans(ir,3)+zshift(ifr),par);
            Hx(ir)=h(1);
            Hy(ir)=h(2);
            Hz(ir)=h(3);
        end
        Nfh=1;
        trans0=zeros(Nfh,3);
        trans0(:,3)=linspace(min(trans(:,3)),max(trans(:,3)),Nfh);
        par0=[ones(Nfh,1)*10 trans0 ones(Nfh,1)*theta(ifr) ...
            ones(Nfh,1)*phi(ifr)];
        h0=complex(zeros(Nfh,3));
        for iz=1:Nfh
            [~,h0(iz,:)]=green3d(freq(ifr),[],0,1,xshift(ifr),...
                yshift(ifr),trans0(iz,3),par0(iz,:));
        end
        h0=real(h0(:,1))*lx(ifr)+real(h0(:,2))*ly(ifr)+...
            real(h0(:,3))*lz(ifr);
        if Nfh>1;h0=interp1(trans0(:,3),h0,trans(:,3));end
        if rc(ifr)<=3
            predData(:,ifr)=(Hx*lx(ifr)+Hy*ly(ifr)+Hz*lz(ifr)-h0)./...
                h0*1e6;
            Ho(:,ifr)=h0;
        else
            predData(:,ifr)=-(Hx*lx(ifr)+Hy*ly(ifr)+Hz*lz(ifr)-h0)./...
                h0*1e6;
            Ho(:,ifr)=-h0;
        end
    end
    predData=predData(:);
    Ho=Ho(:);
    save('invpar.mat','sigb','thb','bnds','xys','dz',...
        'rx','ry','rz','rc','workdir')
else
    [predData,Ho]=getPredAEM(sigb,thb,bnds,xys,dz,sigt,rx,ry,rz,rc,workdir);
end
Frechet=complex(zeros(Nr,Nf,Nc));
%workdir=pwd;
parfor ifr=1:Nf
    trans=[rx-xshift(1)/2 ry-yshift(1)/2 rz-zshift(1)/2];
    rec=[trans(:,1)+xshift(ifr) trans(:,2)+yshift(ifr) ...
        trans(:,3)+zshift(ifr)];
    if ~iter
        [xg,yg,zg]=ndgrid(x,y,z);
        xg=xg(:);yg=yg(:);zg=zg(:);
        Ed=zeros(Nc*3,Nr);
        for ir=1:Nr
            xx=xg-trans(ir,1);
            yy=yg-trans(ir,2);
            zz=zg;
            par=[10 0 0 trans(ir,3) theta(ifr) phi(ifr)];
            [e,~]=green3d(freq(ifr),thb,sigb,...
                ones(size(sigb)),xx,yy,zz,par);
            Ed(:,ir)=e(:);
        end
        G=mgrrecscE(freq(ifr),thb,sigb,ones(size(sigb)),...
           x,xys(1),y,xys(2),z,dz,rec,[4 5 6]);
        subdir=[workdir filesep '0' num2str(ifr)];
        if ~exist(subdir,'dir');mkdir(subdir);end
        binwrite([subdir filesep 'G'],G);
        %G=binread([subdir filesep 'G']);
    else
        subdir=[workdir filesep '0' num2str(ifr)];
        fs=load([subdir filesep 'fwdstg2.mat']);
        Ed=zeros(Nc*3,Nr);
        for ir=1:Nr
            Ed(:,ir)=fs.et.e{ir};
        end
        G=binread([subdir filesep 'G']);
    end
    Ed=reshape(permute(Ed,[2 1]),Nr*Nc,3);
    G=reshape(G(1:Nr,:)*lx(ifr)+G(Nr+1:2*Nr,:)*ly(ifr)+...
        G(2*Nr+1:3*Nr,:)*lz(ifr),Nr*Nc,3);
    Frechet(:,ifr,:)=reshape(sum(G.*Ed,2),Nr,Nc);
end
Frechet=spdiags(1./Ho,0,Nr*Nf,Nr*Nf)*reshape(Frechet,Nr*Nf,Nc)*1e6;

%--------------------------------------------------------------------------
function G=mgrrecscE(f,hh,sig,an,x,dx,y,dy,z,dz,rec,cm)
Nx=length(x); Ny=length(y); Nz=length(z);
Nr=size(rec,1);
cmE=cm(cm<4);Ne=length(cmE);
cmH=cm(cm>3)-3;Nh=length(cmH);
Ncell=Nx*Ny*Nz;
Nxy=Nx*Ny;
xr=rec(:,1)-x(1);
yr=rec(:,2)-y(1);
zr=rec(:,3);
xr1(1,1,:)=xr;
yr1(1,1,:)=yr;
zr1(1,1,:)=zr;
[xx,yy]=ndgrid(dx*(0:Nx-1),dy*(0:Ny-1));
xr2=repmat(xr1,[Nx Ny 1]) - repmat(xx,[1 1 Nr]);
yr2=repmat(yr1,[Nx Ny 1]) - repmat(yy,[1 1 Nr]);
zr2=repmat(zr1,[Nx Ny 1]);
Gxe=zeros(Ne*Nr,Ncell);
Gye=zeros(Ne*Nr,Ncell);
Gze=zeros(Ne*Nr,Ncell);
Gxh=zeros(Nh*Nr,Ncell);
Gyh=zeros(Nh*Nr,Ncell);
Gzh=zeros(Nh*Nr,Ncell);
for ii=1:Nz
    ind=(ii-1)*Nxy+1:ii*Nxy;
    par=[-1 dx dy dz(ii) 0 0 z(ii)];
    [et,ht]=green3d(f,hh,sig,an,xr2,yr2,zr2,par);
    et=et(:,:,:,cmE,:);
    ht=ht(:,:,:,cmH,:);
    e=reshape(et,Nx,Ny,Nr,Ne,3);
    tmpee=reshape(permute(e,[3 4 1 2 5]),Nr*Ne,Nx*Ny,3);
    Gxe(:,ind)=tmpee(:,:,1);
    Gye(:,ind)=tmpee(:,:,2);
    Gze(:,ind)=tmpee(:,:,3);
    e=reshape(ht,Nx,Ny,Nr,Nh,3);
    tmpee=reshape(permute(e,[3 4 1 2 5]),Nr*Nh,Nx*Ny,3);
    Gxh(:,ind)=tmpee(:,:,1);
    Gyh(:,ind)=tmpee(:,:,2);
    Gzh(:,ind)=tmpee(:,:,3);
end
G=[Gxe Gye Gze;Gxh Gyh Gzh];
