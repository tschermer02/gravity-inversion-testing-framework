function [m,dp]=rcgm(do,dp,dInd,m,mapr,mInd,F,Wd,D,Dx,Dy,Dz,logInvInd,Nit,...
    WmCoef,alpIni,alpGIni,betIni,betGIni)
%regularized conjugate gradient method with preconditioner
Wd=getWd(do,dp,Wd,dInd);
Wm=getWm(m,mInd,logInvInd,WmCoef);
[Nd,Nm]=size(F);
F=spdiags(Wd,0,Nd,Nd)*F;
do=Wd.*do;
dp=Wd.*dp;
[x,mltF]=lin2log(m,mapr,Wm,logInvInd);
alp=zeros(Nm,1);alpx=zeros(Nm,1);alpy=zeros(Nm,1);alpz=zeros(Nm,1);
bet=0;betx=0;bety=0;betz=0;
b0=mltF.*(F'*do+alp.*D.^2.*mapr);%problem dependent
b=mltF.*(F'*dp+alp.*D.^2.*m+alpx.*(Dx'*(Dx*m))+alpy.*(Dy'*(Dy*m))+...
    alpz.*(Dz'*(Dz*m)))+bet*GradG(x,x,mInd,speye(Nm,Nm))+...
    betx*GradG(x,x,mInd,Dx)+bety*GradG(x,x,mInd,Dy)+...
    betz*GradG(x,x,mInd,Dz);%problem dependent
r=b-b0;
Precond=ones(size(D));
h=Precond.*r;
del=dot(r,h);
del_b=dot(b0,Precond.*b0);
p=h;
m_1=m;count=0;
misfit=zeros(1,Nit);
rms=zeros(1,Nit);
for it=1:Nit
    if norm(dp-do)/norm(do)<=0.01;break;end
    mltFp=mltF.*p;
    k=dot(p,r)/dot(p,mltF.*(F'*(F*mltFp)+alp.*D.^2.*mltFp+alpx.*(Dx'*(Dx*mltFp))+...
        alpy.*(Dy'*(Dy*mltFp))+alpz.*(Dz'*(Dz*mltFp)))+...
        bet*GradG(p,x,mInd,speye(Nm,Nm))+betx*GradG(p,x,mInd,Dx)+...
        bety*GradG(p,x,mInd,Dy)+betz*GradG(p,x,mInd,Dz));%problem dependent
    dx=-k*p;
    x=x+dx;
    del_1=del;
    p_1=p;
    m=log2lin(x,mapr,Wm,logInvInd);
    dp=dp+F*(mltF.*dx);
    [~,mltF]=lin2log(m,mapr,Wm,logInvInd);
    count=count+1;
    if it==1
      prop=unique(mInd);
      for ip=1:length(prop)
          Dxm=Dx*m;Dym=Dy*m;Dzm=Dz*m;
          im=mInd==prop(ip);id=dInd==prop(ip);
          alp(im)=alpIni{prop(ip)}*norm(do(id)-dp(id))^2/norm(m(im)-mapr(im))^2;
          alpx(im)=alpGIni{prop(ip)}*norm(do(id)-dp(id))^2/norm(Dxm(im))^2;
          alpy(im)=alpGIni{prop(ip)}*norm(do(id)-dp(id))^2/norm(Dym(im))^2;
          alpz(im)=alpGIni{prop(ip)}*norm(do(id)-dp(id))^2/norm(Dzm(im))^2;
      end
      bet=betIni*norm(do-dp)^2/DetG(x,mInd,speye(Nm,Nm));
      betx=betGIni*norm(do-dp)^2/DetG(x,mInd,Dx);
      bety=betGIni*norm(do-dp)^2/DetG(x,mInd,Dy);
      betz=betGIni*norm(do-dp)^2/DetG(x,mInd,Dz);
    elseif any(~(mInd-4))&&(norm(m(mInd==4)-m_1(mInd==4))/...
           norm(m_1(mInd==4))>0.2)
       im=mInd==prop(ip);id=dInd==prop(ip);
       F=spdiags(1./Wd,0,Nd,Nd)*F;dp=dp./Wd;
       mkdir(['IT' num2str(it,'%02d')])
       save(['IT' num2str(it,'%02d') filesep 'invres.mat'],'x',...
           'm','mInd','dp','dInd','misfit','rms','alp','alpx',...
           'alpy','alpz','bet','betx','bety','betz')
       showResults(['IT' num2str(it,'%02d')])
       disp('Updating Frechet ...')
       [Frechet,predData]=updateFrechet(m(im));
       F(id,im)=Frechet;
       dp(id)=predData;
       F=spdiags(Wd,0,Nd,Nd)*F;dp=Wd.*dp;
       m_1=m;count=0;
       alp(im)=min(alp(im)*0.7,alpIni{4}*norm(do(id)-dp(id))^2/...
           norm(m(im)-mapr(im))^2);
    end
    b0=mltF.*(F'*do+alp.*D.^2.*mapr);%problem dependent
    b=mltF.*(F'*dp+alp.*D.^2.*m+alpx.*(Dx'*(Dx*m))+alpy.*(Dy'*(Dy*m))+...
        alpz.*(Dz'*(Dz*m)))+bet*GradG(x,x,mInd,speye(Nm,Nm))+...
        betx*GradG(x,x,mInd,Dx)+bety*GradG(x,x,mInd,Dy)+...
        betz*GradG(x,x,mInd,Dz);%problem dependent
    r=b-b0;
    h=Precond.*r;
    del=dot(r,h);
    del_b=dot(b0,Precond.*b0);
    if it==1;p=h;else p=h+del/del_1*p_1;end
    misfit(it)=norm(dp-do)/norm(do);
    rms(it)=sqrt(del/del_b);
    disp(['It=' num2str(it) ': misfit ' num2str(misfit(it),'%2f')])
end
dp=dp./Wd;
save invres.mat x m mInd dp dInd misfit rms alp alpx alpy alpz bet betx bety betz

%--------------------------------------------------------------------------
function Wd=getWd(do,dp,Wd,dInd)
do=Wd.*do;dp=Wd.*dp;
prop=unique(dInd);
for id=1:length(prop)
    ind=dInd==prop(id);
    Wd(ind)=Wd(ind)/norm(dp(ind)-do(ind))*sqrt(sum(ind));
end

function Wm=getWm(m,mInd,logInvInd,WmCoef)
Wm=zeros(size(m));
prop=unique(mInd);
for ip=1:length(prop);
   ind=mInd==prop(ip);
   Wm(ind)=1/WmCoef{prop(ip)};%problem dependent
end
Wm(logInvInd)=1./log(1./Wm(logInvInd));

%--------------------------------------------------------------------------
function [Frechet,predData]=updateFrechet(sigt)
load('invpar.mat')
[Frechet,predData]=getFrechetAEM(sigb,thb,bnds,xys,dz,sigt,...
    rx,ry,rz,rc,1,workdir);
Frechet=[real(Frechet);imag(Frechet)];
predData=[real(predData);imag(predData)];

%--------------------------------------------------------------------------
function s=GradG(p,m,mInd,Wm)
p=Wm*p;m=Wm*m;
prop=unique(mInd);
if any(~(prop-3))
    Nc=length(m)/4;
    p=reshape(p,Nc,4);
    m=reshape(m,Nc,4);
    if prop(1)==3;p=p(:,[4 1 2 3]);m=m(:,[4 1 2 3]);end
    s=zeros(Nc,4);
    s(:,1)=dot(m(:,2),m(:,2))*p(:,1)-dot(m(:,2),p(:,1))*m(:,2)+...
        dot(m(:,3),m(:,3))*p(:,1)-dot(m(:,3),p(:,1))*m(:,3)+...
        dot(m(:,4),m(:,4))*p(:,1)-dot(m(:,4),p(:,1))*m(:,4);
    s(:,2)=dot(m(:,1),m(:,1))*p(:,2)-dot(m(:,1),p(:,2))*m(:,1);
    s(:,3)=dot(m(:,1),m(:,1))*p(:,3)-dot(m(:,1),p(:,3))*m(:,1);
    s(:,4)=dot(m(:,1),m(:,1))*p(:,4)-dot(m(:,1),p(:,4))*m(:,1);
    ind=find(mInd==3);
    s(:,2)=Wm(ind(1:Nc),ind(1:Nc))'*s(:,2);
    s(:,3)=Wm(ind(Nc+1:2*Nc),ind(Nc+1:2*Nc))'*s(:,3);
    s(:,4)=Wm(ind(2*Nc+1:3*Nc),ind(2*Nc+1:3*Nc))'*s(:,4);
    for ip=1:length(prop)
        if prop(ip)==3;continue;end
        ind=mInd==prop(ip);
        s(:,1)=Wm(ind,ind)'*s(:,1);
    end
    if prop(1)==3;s=s(:,[2 3 4 1]);end
else
    Nc=length(m)/2;
    p=reshape(p,Nc,2);
    m=reshape(m,Nc,2);
    s=zeros(Nc,2);
    s(:,1)=dot(m(:,2),m(:,2))*p(:,1)-dot(m(:,2),p(:,1))*m(:,2);
    s(:,2)=dot(m(:,1),m(:,1))*p(:,2)-dot(m(:,1),p(:,2))*m(:,1);
    for ip=1:length(prop)
       ind=mInd==prop(ip);
       s(:,ip)=Wm(ind,ind)'*s(:,ip);
    end
end
s=s(:);

%--------------------------------------------------------------------------
function s=DetG(m,mInd,Wm)
m=Wm*m;
prop=unique(mInd);
if any(~(prop-3))
    Nc=length(m)/4;
    m=reshape(m,Nc,4);
    if prop(1)==3;m=m(:,[4 1 2 3]);end
    s=dot(m(:,1),m(:,1))*dot(m(:,2),m(:,2))-dot(m(:,1),m(:,2))^2+...
        dot(m(:,1),m(:,1))*dot(m(:,3),m(:,3))-dot(m(:,1),m(:,3))^2+...
        dot(m(:,1),m(:,1))*dot(m(:,4),m(:,4))-dot(m(:,1),m(:,4))^2;
else
    Nc=length(m)/2;
    m=reshape(m,Nc,2);
    s=dot(m(:,1),m(:,1))*dot(m(:,2),m(:,2))-dot(m(:,1),m(:,2))^2;
end

%--------------------------------------------------------------------------
function [m,mltF]=lin2log(sigt,sig0,T,logInvInd)
%m=T*(log(sigt)-log(sigt0)) or m=T*(sigt-sigt0)
m=sigt-sig0;
mltF=ones(size(sigt));
m(logInvInd)=log(sigt(logInvInd)./sig0(logInvInd));
mltF(logInvInd)=sigt(logInvInd);
m=T.*m;mltF=mltF./T;

%--------------------------------------------------------------------------
function sigt=log2lin(m,sig0,T,logInvInd)
%m=T*(log(sigt)-log(sigt0)) or m=T*(sigt-sigt0)
sigt=m./T+sig0;
sigt(logInvInd)=sig0(logInvInd).*exp(m(logInvInd)./T(logInvInd));
