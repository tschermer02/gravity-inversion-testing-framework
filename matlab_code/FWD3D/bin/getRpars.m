function rPars=getRpars(sType,data)
if ischar(data);data=load(data);end
switch sType
    case {1,2,3}%GR&Mag&MagR
        rec=unique(data(:,1:3),'rows');
        rx=rec(:,1);
        ry=rec(:,2);
        rz=rec(:,3);
        rc=unique(data(:,4));
        data=sortrows(data,[1 2 3 4]);
    case 4%AEM(DIGHEM)
        rec=unique(data(:,[3 2 1]),'rows');
        rx=rec(:,3);
        ry=rec(:,2);
        rz=rec(:,1);
        rc=unique(data(:,4));
        data=sortrows(data,[4 3 2 1]);
end
rPars.rx=rx;
rPars.ry=ry;
rPars.rz=rz;
rPars.rc=rc;
rPars.data=data;
