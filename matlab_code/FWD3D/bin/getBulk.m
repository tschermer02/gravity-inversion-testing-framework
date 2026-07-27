function dataBulk=getBulk(sType,rx,ry,rz,rc)
switch sType
    case {1,2,3}%GR&Mag&MagR
        rec=unique([rx(:) ry(:) rz(:)],'rows');
        rc=sort(rc);
        Nr=size(rec,1);
        Ncm=length(rc);
        rx=rec(:,1)';rx=repmat(rx,Ncm,1);
        ry=rec(:,2)';ry=repmat(ry,Ncm,1);
        rz=rec(:,3)';rz=repmat(rz,Ncm,1);
        rc=repmat(rc(:),Nr,1);
        dataBulk=[rx(:) ry(:) rz(:) rc];
    case 4%AEM(DIGHEM)
        rec=unique([rz(:) ry(:) rx(:)],'rows');
        rc=sort(rc);
        Nr=size(rec,1);
        Ncm=length(rc);
        rx=rec(:,3);rx=repmat(rx,Ncm,1);
        ry=rec(:,2);ry=repmat(ry,Ncm,1);
        rz=rec(:,1);rz=repmat(rz,Ncm,1);
        rc=repmat(rc(:)',Nr,1);
        dataBulk=[rx ry rz rc(:)];
end
