function mParsD=interpM(sigb,thb,gPars,mPars,gParsD)
gPars.Nx=length(gPars.x);
gPars.Ny=length(gPars.y);
gPars.Nz=length(gPars.z);
gParsD.Nx=length(gParsD.x);
gParsD.Ny=length(gParsD.y);
gParsD.Nz=length(gParsD.z);
depth=[];
for il=1:length(thb)
    if il==1
        depth(il)=thb(il);
    else
        depth(il)=depth(il-1)+thb(il);
    end
end
mParsD.sigB=[];
Nl=length(thb)+1;
if Nl==1
    mParsD.sigB=repmat(sigb(1,:),length(gParsD.zg),1);
else
    for il=1:Nl
        if il==1
            Nct=sum(gParsD.zg<depth(il));
            mParsD.sigB=[mParsD.sigB;repmat(sigb(il,:),Nct,1)];
        elseif il==Nl
            Nct=sum(gParsD.zg(:)>=mParsD.depth(il-1));
            mParsD.sigB=[mParsD.sigB;repmat(sigb(il,:),Nct,1)];
        else
            Nct=sum(gParsD.zg(:)>=mParsD.depth(il-1)&...
                gParsD.zg(:)<mParsD.depth(il));
            mParsD.sigB=[mParsD.sigB;repmat(sigb(il,:),Nct,1)];
        end
    end
end
mParsD.sigT=mParsD.sigB;
indIn=gParsD.xg>=gPars.x(1)&gParsD.xg<=gPars.x(end)&...
    gParsD.yg>=gPars.y(1)&gParsD.yg<=gPars.y(end)&...
    gParsD.zg>=gPars.z(1)&gParsD.zg<=gPars.z(end);
x=gParsD.xg(indIn);Nx=length(unique(x));
y=gParsD.yg(indIn);Ny=length(unique(y));
z=gParsD.zg(indIn);Nz=length(unique(z));
mPars.sigB=reshape(mPars.sigB,gPars.Nx*gPars.Ny*gPars.Nz,size(sigb,2));
mPars.sigT=reshape(mPars.sigT,gPars.Nx*gPars.Ny*gPars.Nz,size(sigb,2));
mPars.sigA=reshape(mPars.sigA,gPars.Nx*gPars.Ny*gPars.Nz,size(sigb,2));
for idm=1:size(sigb,2);
    if any(indIn)&&Nx>1&&Ny>1&&Nz>1
        sigT=interp3(reshape(gPars.yg,gPars.Nx,gPars.Ny,gPars.Nz),...
            reshape(gPars.xg,gPars.Nx,gPars.Ny,gPars.Nz),...
            reshape(gPars.zg,gPars.Nx,gPars.Ny,gPars.Nz),...
            reshape(mPars.sigT(:,idm),gPars.Nx,gPars.Ny,gPars.Nz),...
            reshape(y,Nx,Ny,Nz),reshape(x,Nx,Ny,Nz),reshape(z,Nx,Ny,Nz));
        mParsD.sigT(indIn,idm)=sigT(:);
    end
end
indNan=isnan(mParsD.sigT);
mParsD.sigT(indNan)=mParsD.sigB(indNan);
mParsD.sigA=mParsD.sigT-mParsD.sigB;
mParsD.sigB=mParsD.sigB(:);
mParsD.sigT=mParsD.sigT(:);
mParsD.sigA=mParsD.sigA(:);
