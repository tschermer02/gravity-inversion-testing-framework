function mPars=getMpars(sigb,thb,gPars,sigt)
%sigb(sigt)-scalar or row vector
depth=[];
for il=1:length(thb)
    if il==1
        depth(il)=thb(il);
    else
        depth(il)=depth(il-1)+thb(il);
    end
end
sigB=[];
Nl=length(thb)+1;
if Nl==1
    sigB=repmat(sigb(1,:),length(gPars.zg),1);
else
    for il=1:Nl
        if il==1
            Nct=sum(gPars.zg<depth(il));
            sigB=[sigB;repmat(sigb(il,:),Nct,1)];
        elseif il==Nl
            Nct=sum(gPars.zg>=depth(il-1));
            sigB=[sigB;repmat(sigb(il,:),Nc,1)];
        else
            Nct=sum(gPars.zg>=depth(il-1)&...
                gPars.zg<depth(il));
            sigB=[sigB;repmat(sigb(il,:),Nct,1)];
        end
    end
end
if isempty(sigt)
    sigT=sigB;
elseif ischar(sigt)
    sigT=load(sigt);
    sigT=reshape(sigT,size(sigB));
else
    sigT=repmat(sigt,length(gPars.zg),1);
end
sigA=sigT-sigB;
mPars.sigB=sigB(:);
mPars.sigT=sigT(:);
mPars.sigA=sigA(:);
