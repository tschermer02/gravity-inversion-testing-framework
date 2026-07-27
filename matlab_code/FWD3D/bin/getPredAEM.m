function [Hp,Ho]=getPredAEM(sigb,thick,bnds,xys,dz,sigT,...
    rx,ry,rz,rc,workdir)
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
theta=theta(rc);
phi=phi(rc);
xshift=xshift(rc);
yshift=yshift(rc);
zshift=zshift(rc);
lx=lx(rc);
ly=ly(rc);
lz=lz(rc);
Nr=length(rx);
Nf=length(freq);
Hp=complex(zeros(Nr,Nf));
Ho=zeros(Nr,Nf);
ts=clock;
%workdir=pwd;
parfor ifr=1:Nf
    trans=[rx-xshift(1)/2 ry-yshift(1)/2 rz-zshift(1)/2];
    rec=[trans(:,1)+xshift(ifr) trans(:,2)+yshift(ifr) ...
        trans(:,3)+zshift(ifr)];
    srcpar=[ones(Nr,1)*10 trans ones(Nr,1)*theta(ifr) ...
        ones(Nr,1)*phi(ifr)];
    subdir=[workdir filesep '0' num2str(ifr)];
    if ~exist(subdir,'dir');mkdir(subdir);end
    cd(subdir)
    delete fwd*.*
    delete sig*.*
    delete int*.*
    delete recp*.*
    runintem(sigb,thick,x,y,z,dz,sigT,rec,srcpar,freq(ifr))
    intout=load('intout.dat');
    cd(workdir)
    Hx=intout(intout(:,4)==4,7:10);
    Hx=complex(Hx(:,1)+Hx(:,3),Hx(:,2)+Hx(:,4));
    Hy=intout(intout(:,4)==5,7:10);
    Hy=complex(Hy(:,1)+Hy(:,3),Hy(:,2)+Hy(:,4));
    Hz=intout(intout(:,4)==6,7:10);
    Hz=complex(Hz(:,1)+Hz(:,3),Hz(:,2)+Hz(:,4));
    Nfh=1;
    trans0=zeros(Nfh,3);
    trans0(:,3)=linspace(min(trans(:,3)),max(trans(:,3)),Nfh);
    srcpar0=[ones(Nfh,1)*10 trans0 ones(Nfh,1)*theta(ifr) ...
        ones(Nfh,1)*phi(ifr)];
    h0=complex(zeros(Nfh,3));
    for iz=1:Nfh
        [~,h0(iz,:)]=green3d(freq(ifr),[],0,1,xshift(ifr),...
            yshift(ifr),trans0(iz,3),srcpar0(iz,:));
    end
    h0=real(h0(:,1))*lx(ifr)+real(h0(:,2))*ly(ifr)+...
        real(h0(:,3))*lz(ifr);
    if Nfh>1;h0=interp1(trans0(:,3),h0,trans(:,3));end
    if rc(ifr)<=3
        Hp(:,ifr)=(Hx*lx(ifr)+Hy*ly(ifr)+Hz*lz(ifr)-h0)./h0*1e6;
        Ho(:,ifr)=h0;
    else
        Hp(:,ifr)=-(Hx*lx(ifr)+Hy*ly(ifr)+Hz*lz(ifr)-h0)./h0*1e6;
        Ho(:,ifr)=-h0;
    end
end
te=clock;
disp(['Elapsed time is ' num2str(etime(te,ts)) ' seconds.'])
Hp=Hp(:);
Ho=Ho(:);

function runintem(sigb,thb,x,y,z,dz,sigT,rec,srcpar,freq)
Ns=size(srcpar,1);
Nf=length(freq);
Nr=size(rec,1);
%--------------------------------------------------------------------------
fid=fopen('intem3d.par','w');
fprintf(fid,'stg=[1 2 3]; \n');
fprintf(fid,'wordy=0; \n');
fprintf(fid,'mfit=1e-6; \n');
fprintf(fid,['sig0=[' num2str(sigb) ']; \n']);
fprintf(fid,['an0=[' num2str(ones(size(sigb))) ']; \n']);
fprintf(fid,['hh0=[' num2str(thb) ']; \n']);
for isc=1:Ns
    fprintf(fid,['srcpar{' num2str(isc) '}=[' ...
        num2str(srcpar(isc,:)) ']; \n']);
end
fprintf(fid,'solflag=6; \n');
fprintf(fid,['x=[' num2str(x) ']; \n']);
fprintf(fid,['y=[' num2str(y) ']; \n']);
fprintf(fid,['z=[' num2str(z) ']; \n']);
fprintf(fid,['dz=[' num2str(dz) ']; \n']);
save sigbody.dat sigT -ascii
fclose(fid);
%--------------------------------------------------------------------------
recpar=zeros(Nf*3*Nr,6);
ibl=0;
for isc=1:Ns
    for icm=4:6
        for ifr=1:Nf
            ibl=ibl+1;
            recpar(ibl,:)=[rec(isc,:) icm isc freq(ifr)];
        end
    end
end
save recpar.dat recpar -ascii
%--------------------------------------------------------------------------
intem3dql
