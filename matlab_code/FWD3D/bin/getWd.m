function Wd=getWd(sType,dataBulk)
switch sType
    case {1,2,3}%GR&Mag&MagR
        Wd=zeros(size(dataBulk,1),1);
        rc=unique(dataBulk(:,4));
        for icm=1:length(rc)
            indc=dataBulk(:,4)==rc(icm);
            doR=dataBulk(indc,5);
            Wd(indc)=1/mean(abs(doR));
        end
    case 4%AEM(DIGHEM)
        rc=unique(dataBulk(:,4));
        Wd=zeros(size(dataBulk,1),1);
        for icm=1:length(rc)
            indc=dataBulk(:,4)==rc(icm);
            doR=dataBulk(indc,5);
            doI=dataBulk(indc,6);
            Wd(indc,1)=1/mean(abs(doR));
            Wd(indc,2)=1/mean(abs(doI));
        end
end
