function showWm(sType,gPars,Wm,barLm,lm3D,verEx,xs,ys,zs,modelDir)
if ~exist(modelDir,'dir');mkdir(modelDir);end
gPars.Nx=length(gPars.x);
gPars.Ny=length(gPars.y);
gPars.Nz=length(gPars.z);
switch sType
    case 1%GR
        logFlag=0;
        flipFlag=0;
        barLbl={''};
        suff={'Dens'};
    case 2%MAG
        logFlag=0;
        flipFlag=0;
        barLbl={''};
        suff={'Sus'};
    case 3%MagR
        Wm=reshape(Wm,gPars.Nx*gPars.Ny*gPars.Nz,3);
        Wm(:,4)=sqrt(Wm(:,1).^2+Wm(:,2).^2+Wm(:,3).^2);
        logFlag=0;
        flipFlag=0;
        barLbl={'','','',''};
        suff={'Mx','My','Mz','Sus'};
    case 4%AEM(DIGHEM)
        logFlag=0;
        flipFlag=0;
        barLbl={''};
        suff={'Res'};
end
for idm=1:size(Wm,2)
    [colmap,cols,lbls]=getColormap(Wm(:,idm),64,11,...
        barLm,logFlag,flipFlag);
    makeAxeSlices(gPars,xs,ys,zs,modelDir,...
        colmap,cols,lbls,verEx,barLbl{idm},suff{idm})
    make3Dview(gPars,Wm(:,idm),lm3D,modelDir,...
        colmap,cols,lbls,verEx,barLbl{idm},suff{idm})
end

%--------------------------------------------------------------------------
function makeAxeSlices(gPars,xs,ys,zs,modelDir,...
    colmap,cols,lbls,verEx,barLbl,suff)
bnds=[gPars.x(1)-gPars.dx/2 gPars.x(end)+gPars.dx/2 ...
    gPars.y(1)-gPars.dy/2 gPars.y(end)+gPars.dy/2 ...
    gPars.z(1)-gPars.dz(1)/2 gPars.z(end)+gPars.dz(end)/2];
Nlbl=length(lbls);
for is=1:length(xs)
    figure
    set(gcf,'visible','off')
    ind=find(abs(gPars.x-xs(is))<=gPars.dx/2);
    if any(ind)
        xComp=gPars.x(ind(end));
        indr=gPars.xg==xComp;
        ycord=[gPars.yg(indr)-gPars.dy/2 ...
            gPars.yg(indr)+gPars.dy/2 ...
            gPars.yg(indr)+gPars.dy/2 ...
            gPars.yg(indr)-gPars.dy/2];
        zcord=[gPars.zg(indr)-gPars.dzg(indr)/2 ...
            gPars.zg(indr)-gPars.dzg(indr)/2 ...
            gPars.zg(indr)+gPars.dzg(indr)/2 ...
            gPars.zg(indr)+gPars.dzg(indr)/2];
        patch2D(ycord,zcord,cols(indr,:));
    end
    axis(bnds([3 4 5 6]))
    xlabel('Y [m]');ylabel('Z [m]');
    set(gca,'Ydir','Reverse')
    daspect([verEx 1 1])
    title(['Vertical section X = ' num2str(xs(is)) ' m'] )
    colormap(colmap);
    h=colorbar;
    CBlim=get(h,'Ylim');
    Ytcks=CBlim(1):diff(CBlim)/(Nlbl-1):CBlim(2);
    set(h,'Ytick',Ytcks,'Yticklabel',num2str(lbls.','%.2f'));
    title(h,barLbl);
    saveas(gcf,[modelDir filesep suff '-Xslice' num2str(is)],'png')
    close(gcf)
end
%--------------------------------------------------------------------------
for is=1:length(ys)
    figure
    set(gcf,'visible','off')
    ind=find(abs(gPars.y-ys(is))<=gPars.dy/2);
    if any(ind)
        yComp=gPars.y(ind(end));
        indr=gPars.yg==yComp;
        xcord=[gPars.xg(indr)-gPars.dx/2 ...
            gPars.xg(indr)+gPars.dx/2 ...
            gPars.xg(indr)+gPars.dx/2 ...
            gPars.xg(indr)-gPars.dx/2];
        zcord=[gPars.zg(indr)-gPars.dzg(indr)/2 ...
            gPars.zg(indr)-gPars.dzg(indr)/2 ...
            gPars.zg(indr)+gPars.dzg(indr)/2 ...
            gPars.zg(indr)+gPars.dzg(indr)/2];
        patch2D(xcord,zcord,cols(indr,:));
    end
    axis(bnds([1 2 5 6]))
    xlabel('X [m]');ylabel('Z [m]');
    set(gca,'Ydir','Reverse')
    daspect([verEx 1 1])
    title(['Vertical section Y = ' num2str(ys(is)) ' m'] )
    colormap(colmap);
    h=colorbar;
    CBlim=get(h,'Ylim');
    Ytcks=CBlim(1):diff(CBlim)/(Nlbl-1):CBlim(2);
    set(h,'Ytick',Ytcks,'Yticklabel',num2str(lbls.','%.2f'));
    title(h,barLbl);
    saveas(gcf,[modelDir filesep suff '-Yslice' num2str(is)],'png')
    close(gcf)
end
%--------------------------------------------------------------------------
for is=1:length(zs)
    figure
    set(gcf,'visible','off')
    ind=find(abs(gPars.z-zs(is))<=gPars.dz/2);
    if any(ind)
        zComp=gPars.z(ind(end));
        indr=gPars.zg==zComp;
        xcord=[gPars.xg(indr)-gPars.dx/2 ...
            gPars.xg(indr)+gPars.dx/2 ...
            gPars.xg(indr)+gPars.dx/2 ...
            gPars.xg(indr)-gPars.dx/2];
        ycord=[gPars.yg(indr)-gPars.dy/2 ...
            gPars.yg(indr)-gPars.dy/2 ...
            gPars.yg(indr)+gPars.dy/2 ...
            gPars.yg(indr)+gPars.dy/2];
        patch2D(xcord,ycord,cols(indr,:));
    end
    axis(bnds([1 2 3 4]))
    xlabel('X [m]');ylabel('Y [m]');
    daspect([1 1 1])
    title(['Horizontal section Z = ' num2str(zs(is)) ' m'] )
    colormap(colmap);
    h=colorbar;
    CBlim=get(h,'Ylim');
    Ytcks=CBlim(1):diff(CBlim)/(Nlbl-1):CBlim(2);
    set(h,'Ytick',Ytcks,'Yticklabel',num2str(lbls.','%.2f'));
    title(h,barLbl);
    saveas(gcf,[modelDir filesep suff '-Zslice' num2str(is)],'png')
    close(gcf)
end

%--------------------------------------------------------------------------
function make3Dview(gPars,m,lims3D,modelDir,...
    colmap,cols,lbls,verEx,dim,typ)
bnds=[gPars.x(1)-gPars.dx/2 gPars.x(end)+gPars.dx/2 ...
    gPars.y(1)-gPars.dy/2 gPars.y(end)+gPars.dy/2 ...
    gPars.z(1)-gPars.dz(1)/2 gPars.z(end)+gPars.dz(end)/2];
Nlbl=length(lbls);
dispnums=false(size(m));
for il=1:size(lims3D,1)
    dispnums=dispnums|(m>=lims3D(il,1)&m<=lims3D(il,2));
end
if any(dispnums)
    figure
    set(gcf,'visible','off')
    xSign=[0 -1 1 0; 0 1 1 0; 0 1 -1 0; 0 -1 -1 0];
    ySign=[0 -1 -1 0; 0 -1 1 0; 0 1 1 0; 0 1 -1 0];
    for k=1:4%top&bottom
        xcord=[gPars.xg(dispnums)+xSign(k,1)*gPars.dx/2 ...
            gPars.xg(dispnums)+xSign(k,2)*gPars.dx/2 ...
            gPars.xg(dispnums)+xSign(k,3)*gPars.dx/2 ...
            gPars.xg(dispnums)+xSign(k,4)*gPars.dx/2];
        ycord=[gPars.yg(dispnums)+ySign(k,1)*gPars.dy/2 ...
            gPars.yg(dispnums)+ySign(k,2)*gPars.dy/2 ...
            gPars.yg(dispnums)+ySign(k,3)*gPars.dy/2 ...
            gPars.yg(dispnums)+ySign(k,4)*gPars.dy/2];
        zcord=repmat(gPars.zg(dispnums)-gPars.dzg(dispnums)/2,1,4);
        patch3D(xcord,ycord,zcord,cols(dispnums,:));
        zcord=repmat(gPars.zg(dispnums)+gPars.dzg(dispnums)/2,1,4);
        patch3D(xcord,ycord,zcord,cols(dispnums,:));
    end
    xSign=[-1 1 1 -1; 1 1 1 1; 1 -1 -1 1; -1 -1 -1 -1];
    ySign=[-1 -1 -1 -1; -1 1 1 -1; 1 1 1 1; 1 -1 -1 1];
    for k=1:4%sides
        xcord=[gPars.xg(dispnums)+xSign(k,1)*gPars.dx/2 ...
            gPars.xg(dispnums)+xSign(k,2)*gPars.dx/2 ...
            gPars.xg(dispnums)+xSign(k,3)*gPars.dx/2 ...
            gPars.xg(dispnums)+xSign(k,4)*gPars.dx/2];
        ycord=[gPars.yg(dispnums)+ySign(k,1)*gPars.dy/2 ...
            gPars.yg(dispnums)+ySign(k,2)*gPars.dy/2 ...
            gPars.yg(dispnums)+ySign(k,3)*gPars.dy/2 ...
            gPars.yg(dispnums)+ySign(k,4)*gPars.dy/2];
        zcord=[gPars.zg(dispnums)-gPars.dzg(dispnums)/2 ...
            gPars.zg(dispnums)-gPars.dzg(dispnums)/2 ...
            gPars.zg(dispnums)+gPars.dzg(dispnums)/2 ...
            gPars.zg(dispnums)+gPars.dzg(dispnums)/2];
        patch3D(xcord,ycord,zcord,cols(dispnums,:));
    end
    axis(bnds)
    xlabel('X [m]');ylabel('Y [m]');zlabel('Z [m]');
    set(gca,'Ydir','Reverse','Zdir','Reverse')
    daspect([verEx verEx 1])
    %title(['3D view of ' typ])
    colormap(colmap)
    h=colorbar;
    CBlim=get(h,'Ylim');
    Ytcks=CBlim(1):diff(CBlim)/(Nlbl-1):CBlim(2);
    set(h,'Ytick',Ytcks,'Yticklabel',num2str(lbls.','%.2f'));
    title(h,dim);
    view(3)
    saveas(gcf,[modelDir filesep typ '-3D'],'png')
    view(0,0)
    saveas(gcf,[modelDir filesep typ '-3Dside'],'png')
    view(0,90)
    saveas(gcf,[modelDir filesep typ '-3Dtop'],'png')
    close(gcf)
end

%--------------------------------------------------------------------------
function [colmap,cols,lbls]=getColormap(m,Nclr,Nlbl,barLm,LogFlag,flipFlag)
colmap=jet;
close(gcf);
if flipFlag;colmap=flipud(colmap);end
if LogFlag
    ind=m<=0;
    m(ind)=barLm(1);
    m=log10(m);
    barLm=log10(barLm);
end
dif=diff(barLm);
colVals=barLm(1):dif/(Nclr-1):barLm(2);
lbls=barLm(1):dif/(Nlbl-1):barLm(2);
if LogFlag;lbls=10.^lbls;end
Nc=length(m);
cols=zeros(Nc,3);
for ic=1:Nc
    [~,colind]=min(abs(m(ic)-colVals));
    cols(ic,:)=colmap(colind(1),:);
end
ind=isnan(m(:,1));
cols(ind,:)=1;

%--------------------------------------------------------------------------
function h=patch2D(vx,vy,col)
[Nc,Np]=size(vx);
vx=vx.';
vy=vy.';
Ver=[vx(:) vy(:)];
Fac=1:Nc*Np;
Fac=reshape(Fac,Np,Nc).';
h=patch('Vertices',Ver,'Faces',Fac,'FaceColor','Flat','FaceVertexCData',col);
set(h,'EdgeColor','none')

%--------------------------------------------------------------------------
function h=patch3D(vx,vy,vz,col)
[Nc,Np]=size(vx);
vx=vx.';
vy=vy.';
vz=vz.';
Ver=[vx(:) vy(:) vz(:)];
Fac=1:Nc*Np;
Fac=reshape(Fac,Np,Nc).';
h=patch('Vertices',Ver,'Faces',Fac,'FaceColor','Flat','FaceVertexCData',col);
set(h,'EdgeColor','none')
