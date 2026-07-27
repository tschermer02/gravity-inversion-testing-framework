function showResults(workdir)
invRes=load([workdir filesep 'invres.mat']);
inpti
verEx=1;
Np=length(bckg);
rPars=cell(1,Np);
mPars=cell(1,Np);
gPars=getGpars(invBounds,xySize,dz);
gParsD=getGpars(dispLims,dispXY,dispDz);
for ip=1:Np
    if ~compFlag(ip);continue;end
    rPars{ip}=getRpars(ip,dataFile{ip});
    mPars{ip}=getMpars(bckg{ip},thick{ip},gPars,[]);
    mPars{ip}.sigT=invRes.m(invRes.mInd==ip);
    mPars{ip}.sigA=mPars{ip}.sigT-mPars{ip}.sigB;
    switch ip
        case {1,2,3}%GR
            dataBulk=[rPars{ip}.data invRes.dp(invRes.dInd==ip)];
        case 4%AEM(DIGHEM)
            dataBulk=[rPars{ip}.data reshape(invRes.dp(invRes.dInd==ip),...
                size(rPars{ip}.data,1),2)];
    end
    save([workdir filesep 'predData' num2str(ip) '.dat'],'dataBulk','-ascii')
    showFields(ip,dataBulk,[workdir filesep 'Fields'])
    mParsD=interpM(bckg{ip},thick{ip},gPars,mPars{ip},gParsD);
    showModel(ip,gParsD,mParsD,barLims{ip},lims3D{ip},...
        verEx,xs,ys,zs,[workdir filesep 'Model']);
end
%invRes.gPars=gPars;invRes.mPars=mPars;
%save([workdir filesep 'invres.mat'],'-struct','invRes')
misfit=invRes.misfit;
save([workdir filesep 'invres.mat'],'misfit','gPars','mPars')
