clc,clear,close all
inpt3
verEx=1;
workDir=[pwd filesep 'work'];
Np=length(bckg);
for ip=1:Np
    if ~compFlag(ip);continue;end
    dataBulk=getBulk(ip,rx{ip},ry{ip},rz{ip},rc{ip});
    rPars=getRpars(ip,dataBulk);
    gPars=getGpars(anomBounds{ip},xySize{ip},dz{ip});
    mPars=getMpars(bckg{ip},thick{ip},gPars,anom{ip});
    switch ip
        case 1%GR
            [xq,yq,zq,w]=getQuadPoints(gPars.xg,gPars.yg,gPars.zg,...
                gPars.dx,gPars.dy,gPars.dzg,1);
            dp=getPredGR(xq,yq,zq,w,mPars.sigT,rPars.rx,...
                rPars.ry,rPars.rz,rPars.rc);
        case 2%Mag
            [xq,yq,zq,w]=getQuadPoints(gPars.xg,gPars.yg,gPars.zg,...
                gPars.dx,gPars.dy,gPars.dzg,1);
            dp=getPredMag(xq,yq,zq,w,mPars.sigT,rPars.rx,...
                rPars.ry,rPars.rz,rPars.rc,Bo,Ao,Io,Do);
        case 3%MagR
            [xq,yq,zq,w]=getQuadPoints(gPars.xg,gPars.yg,gPars.zg,...
                gPars.dx,gPars.dy,gPars.dzg,1);
            dp=getPredMagR(xq,yq,zq,w,mPars.sigT,rPars.rx,...
                rPars.ry,rPars.rz,rPars.rc,Bo,Ao,Io,Do);
        case 4%AEM(DIGHEM)
            dp=getPredAEM(bckg{ip},thick{ip},anomBounds{ip},xySize{ip},...
                dz{ip},mPars.sigT,rPars.rx,rPars.ry,rPars.rz,rPars.rc,workDir);
    end
    if isreal(dp)
        dataBulk=[rPars.data dp];
    else
        dataBulk=[rPars.data real(dp) imag(dp)];
    end
    save(['obsData' num2str(ip) '.dat'],'dataBulk','-ascii')
    showFields(ip,dataBulk,['FieldsP' num2str(ip)]);
    gParsD=getGpars(dispLims,dispXY,dispDz);
    mParsD=interpM(bckg{ip},thick{ip},gPars,mPars,gParsD);
    showModel(ip,gParsD,mParsD,barLims{ip},...
        lims3D{ip},verEx,xs,ys,zs,['ModelP' num2str(ip)]);
end
