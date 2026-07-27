function showFields(sType,databulk,fieldDir)
if~exist(fieldDir,'dir');mkdir(fieldDir);end
units{1}={'mGal','mGal','mGal','mGal','Eotvos','Eotovos',...
    'Eotovos','Eotovos','Eotovos','Eotovos','Eotovos'};
units{2}={'nT','nT','nT','nT'};
units{3}={'nT','nT','nT','nT'};
units{4}={'ppm','ppm','ppm','ppm','ppm'};
suff{1}='GR';
suff{2}='Mag';
suff{3}='Mag';
suff{4}='AEM';
switch sType
    case {1,2,3}%GR&Mag&MagR
        rc=unique(databulk(:,4));
        for icm=1:length(rc)
            indc=databulk(:,4)==rc(icm);
            do=databulk(indc,5);
            if size(databulk,2)>5
                dp=databulk(indc,6);
            else
                dp=[];
            end
            rx=databulk(indc,1);
            ry=databulk(indc,2);
            mHand=makeMaps(do,dp,rx,ry,units{sType}(rc(icm)));
            saveas(mHand,[fieldDir '/' suff{sType} '-C' ...
                num2str(rc(icm))],'png')
        end
    case 4%AEM(DIGHEM)
        rc=unique(databulk(:,4));
        for icm=1:length(rc)
            indc=databulk(:,4)==rc(icm);
            do=databulk(indc,5)+1i*databulk(indc,6);
            if size(databulk,2)>6
                dp=databulk(indc,7)+1i*databulk(indc,8);
            else
                dp=[];
            end
            rx=databulk(indc,1);
            ry=databulk(indc,2);
            mHand=makeMaps(do,dp,rx,ry,units{sType}(rc(icm)));
            saveas(mHand,[fieldDir '/' suff{sType} '-C' ...
                num2str(rc(icm))],'png')
        end
end

%--------------------------------------------------------------------------
function mHand=makeMaps(do,dp,rx,ry,units)
contInt=64;
minx=min(rx);maxx=max(rx);
miny=min(ry);maxy=max(ry);
xs=(maxx-minx)/contInt;
ys=(maxy-miny)/contInt;
xg=minx:xs:maxx;
yg=miny:ys:maxy;
[xg,yg]=meshgrid(xg,yg);
obs=griddata(rx,ry,do,xg,yg);
minRe=min(real(obs(:)));maxRe=max(real(obs(:)));
mHand=figure;
set(mHand,'visible','off')
subplot(2,2,1)
[~,h]=contourf(xg,yg,real(obs),contInt);
colormap(jet)
set(h,'edgecolor','none')
%set(gca,'Ydir','reverse')
xlabel('X [m]')
ylabel('Y [m]')
title('Real observed')
caxis([minRe maxRe]);
hc=colorbar;
set(get(hc,'title'),'string',units)
if ~all(isreal(do))
    minIm=min(imag(obs(:)));maxIm=max(imag(obs(:)));
    subplot(2,2,3)
    [~,h]=contourf(xg,yg,imag(obs),contInt);
    colormap(jet)
    set(h,'edgecolor','none')
    %set(gca,'Ydir','reverse')
    xlabel('X [m]')
    ylabel('Y [m]')
    title('Imag observed')
    caxis([minIm maxIm]);
    hc=colorbar;
    set(get(hc,'title'),'string',units)
end
if ~isempty(dp)
    obs=griddata(rx,ry,dp,xg,yg);
    subplot(2,2,2)
    [~,h]=contourf(xg,yg,real(obs),contInt);
    colormap(jet)
    set(h,'edgecolor','none')
    %set(gca,'Ydir','reverse')
    xlabel('X [m]')
    ylabel('Y [m]')
    title('Real predicted')
    caxis([minRe maxRe]);
    hc=colorbar;
    set(get(hc,'title'),'string',units)
    if ~all(isreal(do))
        minIm=min(imag(obs(:)));maxIm=max(imag(obs(:)));
        subplot(2,2,4)
        [~,h]=contourf(xg,yg,imag(obs),contInt);
        colormap(jet)
        set(h,'edgecolor','none')
        %set(gca,'Ydir','reverse')
        xlabel('X [m]')
        ylabel('Y [m]')
        title('Imag predicted')
        caxis([minIm maxIm]);
        hc=colorbar;
        set(get(hc,'title'),'string',units)
    end
end
