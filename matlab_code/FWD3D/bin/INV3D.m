clc,clear,close all
inpti
verEx=1;
logInvFlag={0,0,0,1};
workDir=[pwd filesep 'work'];
Np=length(bckg);
rPars=cell(1,Np);
mPars=cell(1,Np);
sigT=cell(1,Np);
sigB=cell(1,Np);
Frechet=cell(1,Np);
predData=cell(1,Np);
obsData=cell(1,Np);
Wd=cell(1,Np);
Wm=cell(1,Np);
Delx=cell(1,Np);
Dely=cell(1,Np);
Delz=cell(1,Np);
gPars=getGpars(invBounds,xySize,dz);
gPars.Nx=length(gPars.x);
gPars.Ny=length(gPars.y);
gPars.Nz=length(gPars.z);
Nd=zeros(1,Np);
Nm=zeros(1,Np);
for ip=1:Np
    if ~compFlag(ip);continue;end
    rPars{ip}=getRpars(ip,dataFile{ip});
    mPars{ip}=getMpars(bckg{ip},thick{ip},gPars,[]);
    sigT{ip}=mPars{ip}.sigT;
    sigB{ip}=mPars{ip}.sigB;
    Delx{ip}=grad3d(gPars.Nx,gPars.Ny,gPars.Nz,1);
    Dely{ip}=grad3d(gPars.Nx,gPars.Ny,gPars.Nz,2);
    Delz{ip}=grad3d(gPars.Nx,gPars.Ny,gPars.Nz,3);
    switch ip
        case 1%GR
            obsData{ip}=rPars{ip}.data(:,5);
            [xq,yq,zq,w]=getQuadPoints(gPars.xg,gPars.yg,gPars.zg,...
                gPars.dx,gPars.dy,gPars.dzg,1);
            [Frechet{ip},predData{ip}]=getFrechetGR(xq,yq,zq,w,mPars{ip}.sigT,...
                rPars{ip}.rx,rPars{ip}.ry,rPars{ip}.rz,rPars{ip}.rc);
            dataBulk=[rPars{ip}.data predData{ip}];
        case 2%Mag
            obsData{ip}=rPars{ip}.data(:,5);
            [xq,yq,zq,w]=getQuadPoints(gPars.xg,gPars.yg,gPars.zg,...
                gPars.dx,gPars.dy,gPars.dzg,1);
            [Frechet{ip},predData{ip}]=getFrechetMag(xq,yq,zq,w,mPars{ip}.sigT,...
                rPars{ip}.rx,rPars{ip}.ry,rPars{ip}.rz,rPars{ip}.rc,Bo,Ao,Io,Do);
            dataBulk=[rPars{ip}.data predData{ip}];
        case 3%MagR
            obsData{ip}=rPars{ip}.data(:,5);
            [xq,yq,zq,w]=getQuadPoints(gPars.xg,gPars.yg,gPars.zg,...
                gPars.dx,gPars.dy,gPars.dzg,1);
            [Frechet{ip},predData{ip}]=getFrechetMagR(xq,yq,zq,w,mPars{ip}.sigT,...
                rPars{ip}.rx,rPars{ip}.ry,rPars{ip}.rz,rPars{ip}.rc,Bo,Ao,Io,Do);
            dataBulk=[rPars{ip}.data predData{ip}];
            Delx{ip}=blkdiag(Delx{ip},Delx{ip},Delx{ip});
            Dely{ip}=blkdiag(Dely{ip},Dely{ip},Dely{ip});
            Delz{ip}=blkdiag(Delz{ip},Delz{ip},Delz{ip});
        case 4%AEM(DIGHEM)
            obsData{ip}=[rPars{ip}.data(:,5);rPars{ip}.data(:,6)];
            [Frechet{ip},predData{ip}]=getFrechetAEM(bckg{ip},thick{ip},...
                invBounds,xySize,dz,mPars{ip}.sigT,rPars{ip}.rx,...
                rPars{ip}.ry,rPars{ip}.rz,rPars{ip}.rc,0,workDir);
            dataBulk=[rPars{ip}.data real(predData{ip}) imag(predData{ip})];
            Frechet{ip}=[real(Frechet{ip});imag(Frechet{ip})];
            predData{ip}=[real(predData{ip});imag(predData{ip})];
    end
    [Nd(ip),Nm(ip)]=size(Frechet{ip});
    Wd{ip}=getWd(ip,dataBulk);
    Wd{ip}=spdiags(Wd{ip}(:),0,Nd(ip),Nd(ip));
    Wm{ip}=getWm(Wd{ip}*Frechet{ip});%depth weighting
    %showWm(ip,gPars,Wm{ip},[0 1],[],verEx,xs,ys,zs,'Sens');
    Wm{ip}=spdiags(Wm{ip},0,Nm(ip),Nm(ip));
end
if sum(compFlag)==2%joint inversion
    m=[];mapr=[];do=[];dp=[];
    logInvInd=logical([]);mInd=[];dInd=[];
    for ip=1:Np
        if ~compFlag(ip);continue;end
        m=[m;sigT{ip}];
        mapr=[mapr;sigB{ip}];
        do=[do;obsData{ip}];
        dp=[dp;predData{ip}];
        logInvInd=[logInvInd;logical(ones(Nm(ip),1)*logInvFlag{ip})];
        mInd=[mInd;ones(Nm(ip),1)*ip];
        dInd=[dInd;ones(Nd(ip),1)*ip];
    end
    F=blkdiag(Frechet{compFlag>0});
    Dx=blkdiag(Delx{compFlag>0});
    Dy=blkdiag(Dely{compFlag>0});
    Dz=blkdiag(Delz{compFlag>0});
    [m,dp]=rcgm(do,dp,dInd,m,mapr,mInd,F,full(diag(blkdiag(Wd{compFlag>0}))),...
        full(diag(blkdiag(Wm{compFlag>0}))),Dx,Dy,Dz,logInvInd,Nit,...
        WmCoef,alpIni,alpGIni,betIni,betGIni);
    for ip=1:Np
        if ~compFlag(ip);continue;end
        sigT{ip}=m(mInd==ip);
        predData{ip}=dp(dInd==ip);
    end
end
gParsD=getGpars(dispLims,dispXY,dispDz);
for ip=1:Np
    if ~compFlag(ip);continue;end
    mPars{ip}.sigT=sigT{ip};
    mPars{ip}.sigA=mPars{ip}.sigT-mPars{ip}.sigB;
    switch ip
        case {1,2,3}%GR/MAG/MAGR
            dataBulk=[rPars{ip}.data predData{ip}];
        case 4%AEM(DIGHEM)
            dataBulk=[rPars{ip}.data reshape(predData{ip},...
                size(rPars{ip}.data,1),2)];
    end
    if ~exist('Final');mkdir('Final');end
    save(['Final' filesep 'predData' num2str(ip) '.dat'],'dataBulk','-ascii')
    showFields(ip,dataBulk,['Final' filesep 'Fields'])
    mParsD=interpM(bckg{ip},thick{ip},gPars,mPars{ip},gParsD);
    showModel(ip,gParsD,mParsD,barLims{ip},lims3D{ip},...
        verEx,xs,ys,zs,['Final' filesep 'Model']);
end
invRes=load('invres.mat');
%invRes.gPars=gPars;invRes.mPars=mPars;
%save(['Final' filesep 'invres.mat'],'-struct','invRes');
misfit=invRes.misfit;
save(['Final' filesep 'invres.mat'],'misfit','gPars','mPars');
delete invpar.mat invres.mat
