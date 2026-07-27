function intem3dql(varargin)
% INTEM3DQL
% This code has been modified to enable to compute QL approximation based
% on INTEM3D3 (GMRM bug fix version of INTEM3D)
% see also INTEM3D
%
% INTEM3DQL 3-D Electromagnetic Forward Modeling Based on IE method
%
% Type
%
%     intem3dql -h
%
% for more information.
%

%
% Last modified: 2006/03/08 11:47:57 MST
%
%  developed by Gabar Hursan and Michael Zhdanov (mzhdanov@mines.utah.edu)
% maintained by Takumi Ueda (tueda@mines.utah.edu)
%

% check and read inout argment(stg)
if ((nargin>0)&(length(varargin{1})<2)); error('  Invalid input argument'); end;

if (nargin==1)&(varargin{1}(1:2)=='-h')
  disphelp;
  return;
else
  p={' ',' '};
  p=buf2par(p,arg2buf(nargin,varargin));
  p=buf2par(p,readtxt('intem3d.par'));
end

% check for required subfunctions
% checksubfunc;

if (nargin==1)&(varargin{1}(1:2)=='-h')
  disphelp;
  return;
end

%--------------------------------------------------------------------------
%  Read "intem3d.par"
%--------------------------------------------------------------------------
[srcpar,sig0,hh0,an0,x,y,z,dz,             ...
      kcomp,sigc0,chrgabl0,timeconst0,freqconst0,  ...
      mfit, solflag, stg, wordy, FNxy, tmpsave, ...
      tth, quickEn, swabse, lqlint, Nsubset,fastEn, subtx, combo] ...
    =  read_intem3d_par(nargin,varargin,p);

save inputpara ;  % Input parameters are saved "inputpara.mat"

mosvcp('sigbody.dat','sigbody_tmp.dat');

%--------------------------------------------------------------------------
%  Calculation of conductivity of background layered earth.
%--------------------------------------------------------------------------
%   For kcomp(1)=0 or 1, sig0 is a vector containing conductivity of each
%   layer. (Same as previous version)
%   For kcomp(1)=2, sig0 is a cell array of vectors containing
%   conductivity of each layer with frequencies shown in "recpar.dat"
%--------------------------------------------------------------------------
sig0 = get_background_layer_sig(kcomp,sig0,sigc0,chrgabl0,timeconst0,freqconst0);

% prepare 1D background parameters
sigset.real = sig0;
sigset.imag = sigc0;
sigset.chrg = chrgabl0;
sigset.timc = timeconst0;
sigset.frqc = freqconst0;

% checking MATLAB version
[mlversion] = chkmlver;

x  = str2num(sprintf('%12.9e ', x));
y  = str2num(sprintf('%12.9e ', y));
z  = str2num(sprintf('%12.9e ', z));
dz = str2num(sprintf('%12.9e ', dz));

% generate *f (for fine grid) as original finer grid data;
xf=x; yf=y; zf=z; dzf=dz;

% new subfunction for preparation of 3D anomalous conductivity info.
%prep_sigbody(x,y,z,dz);

if solflag >= 7;
  % Only for the case of 'solflag>=7' (dual grid QL or LQL approx.)
  % prepare coase grid with user input combination parameter
  [xc,yc,zc,dzc]=prep_input_pars_ql(x,y,z,dz,combo) ;
  solg=2;

else

% Born, QA and Full IEs
  solg=1;

end

switch solg
  case 1
    [err] = prep_sigbody(hh0,sigset,kcomp,x,y,z,dz);
  case 2
    [err] = prep_sigbody(hh0,sigset,kcomp,x,y,z,dz,xc,yc,zc,dzc);
end

%---------------------------------------------------------------------------------------
% start main computation
% stage 1 : normal E fields at receivers
%       2 : total E fields inside the anomalous domain
%       3 : total E fields at receivers
%---------------------------------------------------------------------------------------

for key = stg

  if(wordy) fprintf(1,'\n Stage %g.\n',key); end;

  switch(key)

   case 0; if(wordy);  disp('Test preparation'); end;

   case 1;

    t = cputime ;
    for igrid = 1:solg;
      if solflag >= 7 ; % QL or LQL
        if igrid ==1;
          if wordy >=2 ;disp(['Preparation for finer / regular grid']); end;
          x=xf;y=yf;z=zf;dz=dzf;
          if exist('sigbodyf.dat')==2;
            mosvcp('sigbodyf.dat','sigbody_tmp.dat');
          end

        elseif igrid==2;
          if wordy >=2 ; disp(['Preparation for coarse grid (QL / LQL)']); end;
          x=xc;y=yc;z=zc;dz=dzc;
          if exist('sigbodyc.dat')==2;
            mosvcp('sigbodyc.dat','sigbody_tmp.dat');
          end
        end
      end
      if(wordy); disp('checking of inputs, computing background fld. at the receivers'); end;
      [x,y,z,dz]= inptchk(hh0,sig0,an0,x,y,z,dz);
      recparnm='recpar.dat';
      if (exist(recparnm)==2)
        tmp=loadasc(recparnm);
      else
        error('recpar.dat does not exist.');
      end

      tmp=loadasc(recparnm);
      [x1,y1,indbody] = datachk(tmp,srcpar,x,y,z);
      % intem3d orignal
      if subtx == 1
        [xyd,dn,f,zr,dind,srcparreal,indsrt,indinvsrt] = dataprep(wordy,tmp,srcpar,hh0,sig0,an0);
      else
        % modification for sub txs
        [xyd,dn,f,zr,dind,srcparreal,indsrt,indinvsrt] = dataprep_subtx(wordy,tmp,srcpar,hh0,sig0,an0,subtx);
      end

      % space estimation
      if wordy>1; spcest(x,y,z,x1,y1,xyd,solflag); end;
      if(wordy); disp('computing interpolation matrices for the receivers'); end;
      wi = fillwi(x1,y1,xyd);
      tsfilenm='sigbody_tmp.dat';

      if (exist(tsfilenm)==2);
        % compute sigma for 3D body
        sigtot=loadascc(tsfilenm,kcomp,f);
      else
        disp('sigbody.dat does not exist.'); error('Please check your sbody.dat.');
      end

      Nxyz = length(x) * length(y) * length(z);
      if length(sigtot)==1 ; sigtot = sigtot * ones(Nxyz,1); end
      if ( (length(sigtot)~=Nxyz) & (length(sigtot)~=1) )
        error(['There should be one or ' num2str(Nxyz) ' conductivity values specified in sigbody.dat.']);
      end

      % multi frequencies version from Dr. Lee [2004-11-09]
      sigbg = bgcond_ip(hh0,sig0,x,y,z);
      ds = cal_del_sig(sigtot,sigbg);

      save tmpall ;
      copy_pars_as_dual_grid;
      clear tmpall;

    end

    t1 = cputime - t ;
    if(wordy);
      fprintf('\n\n\nSTAGE 1 : %g [sec]\n\n\n',t1);
    end

    % stage 1 end.

    % ----------
    % STAGE 2
    % ----------

   case 2;

    t = cputime ;
    if(wordy);
      disp('Calculating electric fields inside the anomalous domain');
    end;

    switch solflag
     case {7,8,9,10}
      load fwdstg1c.mat;
      et = fillets(wordy,f,hh0,sig0,an0,xc,yc,zc,dzc,FNxyc,dsc,sigbgc,mfit,solflag,srcparrealc,tmpsave,tth,quickEn,swabse,lqlint,Nsubset,fastEn,subtx);
     otherwise
      load fwdstg1.mat;
      et = fillets(wordy,f,hh0,sig0,an0,x,y,z,dz, FNxy,ds,sigbg,mfit,solflag,srcparreal,tmpsave,tth,quickEn,swabse,lqlint,Nsubset,fastEn,subtx);
    end

    if wordy>=1 ; fprintf('Now saving the result of stage 2\n'); end ;

    save fwdstg2.mat et;

    if(wordy);
      fprintf('... done as fwdstg2.mat\n');
      t2 = cputime - t ;
      fprintf('\n\n\nSTAGE 2 : %g [sec]\n\n\n',t2);
    end

    % ----------
    % STAGE 3
    % ----------

   case 3; if(wordy); disp('Calculating anomalous fields at the receivers'); end;

    t = cputime ;
    load fwdstg1.mat;
    load fwdstg2.mat;

    if solflag ~= -1
      save etstage2 et ;
    end

    if(wordy);
      fprintf('\n Computing FWDREC_IP (Stage 3)\n\n')
    end

    da = fwdrec_ip(wordy, f, hh0, sig0, an0, x1, y1, z, dz, FNxy, zr, dind, indbody, ds, et, wi);

    if(wordy);fprintf('\n Sorting anomalous field (Stage 3)\n\n');end
    da = da(indinvsrt,:);

    if(wordy);fprintf('\n Sorting background field (Stage 3)\n\n');end
    dn = dn(indinvsrt,:);

    tmp=loadasc('recpar.dat');
    recpar=tmp;
    subtx = 1;

    if(wordy);fprintf('\n Checking duplicated field (Stage 3)\n\n');end;

    dacorr = corrfld(recpar,et,x,y,z,dz,da,dn,subtx);

    printout(recpar,dacorr,dn,subtx);

    if exist('sigbodyf.dat','file') == 2
      delete('sigbodyf.dat') ;
    end
    if exist('sigbodyc.dat','file') == 2
      delete('sigbodyc.dat') ;
    end
    if exist('sigbody_tmp.dat','file') == 2
      delete('sigbody_tmp.dat') ;
    end


    % clean up working directory
    if exist('etstage2.mat'); delete etstage2.mat; end;
    if exist('tmpall.mat'); delete tmpall.mat; end;
    if exist('inputpara.mat'); delete inputpara.mat; end;

    t3 = cputime - t ;
    if(wordy);fprintf('STAGE 3 : %g [sec]\n',t3);end;

  end

end

if(wordy);
  fprintf('--------------------------\n');
  fprintf('CPUTIME for STAGE 1 to 3 (temp)\n\n');
  if exist('t1')
    fprintf('STAGE 1 : %g [sec]\n',t1);
  end
  if exist('t2')
    fprintf('STAGE 2 : %g [sec]\n',t2);
  end
  if exist('t3')
    fprintf('STAGE 3 : %g [sec]\n',t3);
  end
  fprintf('--------------------------\n');
end


% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function buf=arg2buf(n,v)
% convert arguments
% into a character buffer
%

buf=[];
for k=1:n
 if(ischar(v{k}))
  buf=[buf v{k} sprintf('\n')];
 end
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function sb = bgcond(h,sig,x,y,z)

% ----------------------------------------------
% Computing background conductivity in the mesh
% ----------------------------------------------
%
% h       : vector of layer thicknesses, [] for halfspace
% sig     : vector of layer conductivities
% x	  : vector of the x coordinates of the mesh
% y	  : vector of the y coordinates of the mesh
% z	  : vector of the z coordinates of the mesh
% sb	  : Nxyz,1 vector of background conductivity

h = (h(:))';
x=length(x); Ny=length(y); Nz=length(z);
Nxyz=Nx*Ny*Nz;
%nl = length(sig);
nl = length(h)+1;

h=[0 h inf];

zbnd=zeros(1,length(h));
zbnd(1)=h(1);
for i=2:length(h)
   zbnd(i)=zbnd(i-1)+h(i);
end

sb = zeros(0,1);
for i=1:nl
   ind = find((z>zbnd(i)) & (z<zbnd(i+1)));
   numc=length(ind)*Nx*Ny;
   sb=[sb;sig(i)*ones(numc,1)];
end


% ----------------------------------------------------------------------------
function sb = bgcond_ip(h,sigf,x,y,z)
% ----------------------------------------------------------------------------
% Computing background conductivity in the mesh
%
% h       : vector of layer thicknesses, [] for halfspace
% sig     : vector of layer conductivities
% x	  : vector of the x coordinates of the mesh
% y	  : vector of the y coordinates of the mesh
% z	  : vector of the z coordinates of the mesh
% sb	  : Nxyz,1 vector of background conductivity
if (iscell(sigf))
  kcomp = 2;
  nfrq = length(sigf);
else
  kcomp = 0;
  sig = sigf;
end

h = (h(:))';
Nx=length(x); Ny=length(y); Nz=length(z);
Nxyz=Nx*Ny*Nz;
%nl = length(sig);
nl = length(h)+1;

h=[0 h inf];
zbnd=zeros(1,length(h));
zbnd(1)=h(1);
for i=2:length(h)
   zbnd(i)=zbnd(i-1)+h(i);
end

sb = zeros(0,1);
if (kcomp ~= 2)
  for i=1:nl
    ind = find((z>zbnd(i)) & (z<zbnd(i+1)));
    numc=length(ind)*Nx*Ny;
    sb=[sb;sig(i)*ones(numc,1)];
  end
else
  sb = cell(1,nfrq);
  for ifrq = 1:nfrq
    sig = sigf{ifrq};
    sbtmp = zeros(0,1);
    for i=1:nl
      ind = find((z>zbnd(i)) & (z<zbnd(i+1)));
      numc=length(ind)*Nx*Ny;
      sbtmp=[sbtmp;sig(i)*ones(numc,1)];
    end
    sb{ifrq} = sbtmp;
  end
end



% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function x = bicgstgab(wordy,vb,ds3,b,m1,m2,x0,phi0);
%
% x = bicgstgab(wordy,vb,ds3,b,m1,m2,x0,phi0);
%
% BiCGSTAB for forward problem
%
% wordy - if wordy>1:display errors; if wordy<1: no messages
% A	- coefficient matrix
% b	- right hand side
% m1	- left preconditioner matrix
% m2	- right preconditioner matrix
% x0	- initial guess
% phi0	- stopping error (relative residual)
% s	- number of inner iterations

Nm = length(ds3);
%Nm = size(A,1);
if length(m1)==0; m1=ones(Nm,1); end;
if length(m2)==0; m2=ones(Nm,1); end;

% ----- integral equation stuff

m1m2 = m1.*m2;
sam2 = ds3.*m2;
b  = m1.*b;
x  = x0./m2;
r  = x.*m1m2 - m1.*multgb1(vb,sam2.*x) - b; % r=M1*A*M2*x-M1*b

% -------------------------------------

mf0 = b'*b;

rst = r; beta = 0; p=0;Ap=0;psi=0;
tic;
for it=1:Nm*2

  mfit = abs(r'*r);
  err  = sqrt(mfit/mf0);
  if(wordy>1); displayfwdmfit(it,err,toc); end;
  if(err<phi0) break; end;

  p  = r + beta*(p - psi*Ap);

%  Ap = A*p;
  Ap	= p.*m1m2 - m1.*multgb1(vb,sam2.*p);

  alpha = (rst'*r)/(rst'*Ap);

  t = r - alpha*Ap;
%  At = A*t;
  At	= t.*m1m2 - m1.*multgb1(vb,sam2.*t);

  psi = (At'*t)/(At'*At);
  x  = x - alpha*p - psi*t;

  rold = r;
  r = t - psi*At;

  beta = (alpha/psi)*(rst'*r)/(rst'*rold);

end

x=x.*m2;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function w=blin2dm(x,y,xy)
% BLIN2DM regular->irregular 2d interpolation matrix
%
% computes sparse 2-D bilinear
% interpolation matrix
% from regular to irregular grid.
%
% w=blin2dm(x,y,xy)
%
% w - resulting interpolation matrix
% x,y - nodes of regular grid, x dimension fastest in w
% xy - irregular grid, xy(:,1) : x-components
%

% 99Dec14 ONP
%
% 1----------- -----------2    -1  iy-1
% |           |           |
% |           |           |
% |     *     |y          |
% |           |           |
% |           |           |
%  -----------0-----------      0
% |     x     |           |
% |           |           |
% |           |           |
% |           |           |
% |           |           |
% 3----------- -----------4     1  iy
%
% -1          0           1  x  y
% ix-1                   ix
%
% p(x,y)= [ { p1*(1-x) + p2*(1+x) }*(1-y) +
%           { p3*(1-x) + p4*(1+x) }*(1+y)   ]/4
%
% result=[1 y x xy]*abcd*[p1 p2 p3 p4]';
%

% col=inline('[[1 d] b*[1 d]]','b','d');
%
abcd=0.25*[col( -1, -1)'...  % p1
           col(  1, -1)'...  % p2
           col( -1,  1)'...  % p3
           col(  1,  1)'];   % p4

% check points outside the model domain

if( (max(xy(:,1))>max(x)) + (min(xy(:,1))<min(x)) + ...
    (max(xy(:,2))>max(y)) + (min(xy(:,2))<min(y))  )
 error('There are points outside the domain');
 return;
end


%Nxy=length(xy);
Nxy=size(xy,1);
Ny=length(y);
Nx=length(x);

dx2=0.5*(x(2)-x(1));
dy2=0.5*(y(2)-y(1));

w=spalloc(Nxy,Nx*Ny,Nxy*4); % allocate for speed

for iy=2:Ny

 y1=y(iy-1);
 y2=y(iy);
 iny=[iy-1 iy-1  iy   iy];

 jy=find( (xy(:,2)>=y1).*(xy(:,2)<=y2) );

 for ix=2:Nx

  x1=x(ix-1);
  x2=x(ix);

  in=jy( find((xy(jy,1)>=x1).*(xy(jy,1)<=x2)) );
  Nin=length(in);

  if(Nin>0)
   xp=(xy(in,1)-(x1+x2)*0.5)/dx2;
   yp=(xy(in,2)-(y1+y2)*0.5)/dy2;
   inx=[ix-1 ix    ix-1 ix];
   w(in,inx+(iny-1)*Nx)=[ones(Nin,1) yp xp xp.*yp]*abcd;

%   plot(xp*dx2+(x1+x2)*0.5,yp*dy2+(y1+y2)*0.5,'r*');
%   hold on;
%   plot([x1 x1],[y1 y2],'k*',[x2 x2],[y1 y2],'g*');
%   size(w)

  end

 end
end


[i1,i2,v] = find(w);
w=sparse(i1,i2,v,Nxy,Nx*Ny);

return

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function par=buf2par(par,buf)
% convert character buffer
% to a structure array
%

if(isempty(par)) return; end;
lf=sprintf('\n');
ind=[];
if(isempty(buf)==0)
 ind=find((buf==lf)|(buf==';'));
end

if(isempty(ind)==0) buf(ind)=' '; end;

ind=[1 ind length(buf)];
ii=size(par,1);

for k=1:length(ind)-1
   str=buf(ind(k):ind(k+1));
   in=min(find(str=='%'));
   if(length(in)>0) str=str(1:in-1); end;
   in=find(str=='=');
   if(length(in)>0)
      ii=ii+1;
      par{ii,1}=delspace(str(1:in-1));
      par{ii,2}=str(in+1:end);
   end
end

return

function res=delspace(str)

 in=find(str~=' ');
 if(length(in)>0) res=str(in); end;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function res=col(b,d)

% instead of inline of col
% for compiler compatibility
%
 res=[[1 d] b*[1 d]];
 return


% ----------------------------------------------------------------------------
function ds = cal_del_sig(sigtot,sigbg)
% ----------------------------------------------------------------------------
if (iscell(sigbg))
  kcomp1 = 2;
  nf1 = length(sigbg);
else
  kcomp1 = 0;
end
if (iscell(sigtot))
  kcomp2 = 2;
  nf2 = length(sigtot);
else
  kcomp2 = 0;
end

if (kcomp1 == 0) & (kcomp2 == 0)
  ds = sigtot-sigbg;
elseif (kcomp1 == 2) & (kcomp2 == 2)
  for ifrq=1:nf1
    ds{ifrq} = sigtot{ifrq}-sigbg{ifrq};
  end
elseif (kcomp1 == 0) & (kcomp2 == 2)
  for ifrq=1:nf2
    ds{ifrq} = sigtot{ifrq}-sigbg;
  end
elseif (kcomp1 == 2) & (kcomp2 == 0)
  for ifrq=1:nf1
    ds{ifrq} = sigtot-sigbg{ifrq};
  end
end




% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function dacorr = corrfld(recpar,et,x,y,z,dz,da,dn,subtx);

Nx = length(x); Ny = length(y); Nz = length(z);
Nd = size(recpar,1);
% modification for subtx [2004-12-06]
Nd = Nd / subtx ;
ind = [1:Nd];

xr = recpar(ind,1);
yr = recpar(ind,2);
zr = recpar(ind,3);
ci = recpar(ind,4);
si = recpar(ind,5);
fi = recpar(ind,6);

srci = unique(si); Ns = length(srci);

for ii = 1:Ns; si( find(si==srci(ii)) )=ii; end

%frqi = unique(fi); Nf = length(frqi);
%for ii = 1:Nf; fi( find(fi==frqi(ii)) )=ii; end

%% bug fix by Nick 2004-0907
%Change these lines !
frqi = unique(fi); Nf = length(frqi);
for ii = 1:Nf; fi1( find(fi==frqi(ii)) )=ii; end
fi=fi1.';
%---------------


dx = (x(2)-x(1))*ones(Nx,1); dy = (y(2)-y(1))*ones(Ny,1);

xi = zeros(Nd,1); yi = zeros(Nd,1); zi = zeros(Nd,1);

for ii = 1:Nx; xi( find( abs(xr-x(ii))<=dx(ii)/2 ) )=ii; end
for ii = 1:Ny; yi( find( abs(yr-y(ii))<=dy(ii)/2 ) )=ii; end
for ii = 1:Nz; zi( find( abs(zr-z(ii))<=dz(ii)/2 ) )=ii; end

dacorr=da;
%indcorr=find( (xi~=0)&(yi~=0)&(zi~=0)&(ci<4) )
% subtx bug fix ... not perfect... FIX ASAP
indcorr=find( (xi~=0)&(yi~=0)&(zi~=0)&(ci<4) );

if any(indcorr)
  xinin=xi(indcorr);
  yinin=yi(indcorr);
  zinin=zi(indcorr);
  cinin=ci(indcorr);
  sinin=si(indcorr);
  finin=fi(indcorr);

  etall = zeros(Nx,Ny,Nz,3,Nf,Ns);
  for ii = 1:Nf
    for jj = 1:Ns
      etall(:,:,:,:,ii,jj)=reshape(et.e{ii,jj},Nx,Ny,Nz,3);
    end
  end
  etall=etall(:);
  iiii = sub2ind([Nx Ny Nz 3 Nf Ns],xinin,yinin,zinin,cinin,finin,sinin);
  dt = etall(iiii);
  dacorr(indcorr)=dt-dn(indcorr);
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [x1,y1,indbody] = datachk(inptdat,srcpar,x,y,z);

Nx=length(x); Ny=length(y);Nz=length(z);

Nrow = size(inptdat,1);
Ncol = size(inptdat,2);

errind = 0;

% ---- Checking if there are at least 2 cells in x and y directions. ---

if(Nx==1);
  error('There must be at least 2 cells in x direction.');
else;
  dx=x(2)-x(1);
end

if(Ny==1);
  error('There must be at least 2 cells in y direction.');
else;
  dy=y(2)-y(1);
end


% ---- Checking the right number of columns in the input data file ---

if Ncol ~= 6
  disp(['There are ' num2str(Ncol) ' columns in recpar.dat. ']);
  disp(['The correct number of columns is 6.']);
  error(' ');
end



% ---- checking for data points outside the domain and
% ---- producing larger domain (x1,y1) covering all
% ---- receivers horizontally


xr = inptdat(:,1); yr = inptdat(:,2);
xrmin = min(xr); yrmin = min(yr);
xrmax = max(xr); yrmax = max(yr);

x1=x(:);
  kxmin=max([0 ceil( (x(1)-xrmin)/dx )]);
  addxmin=x(1)-dx*(kxmin:-1:1);
  x1=[addxmin(:);x1];
  kxmax=max([0 ceil( (xrmax-x(end))/dx )]);
  addxmax=x(end)+dx*(1:kxmax);
  x1=[x1;addxmax(:)];
y1=y(:);
  kymin=max([0 ceil( (y(1)-yrmin)/dy )]);
  addymin=y(1)-dy*(kymin:-1:1);
  y1=[addymin(:);y1];
  kymax=max([0 ceil( (yrmax-y(end))/dy )]);
  addymax=y(end)+dy*(1:kymax);
  y1=[y1;addymax(:)];

Nx1=length(x1); Ny1=length(y1);

ixrin=kxmin+(1:Nx);
iyrin=kymin+(1:Ny);
[ix,iy,iz]=ndgrid(ixrin,iyrin,1:Nz);
indbody = sub2ind([Nx1 Ny1 Nz],ix(:),iy(:),iz(:));
indbody=indbody(:);


% ---- checking for data points having the same location,
%      component and source indices and frequency ----

[tmp,indi,indj]=unique(inptdat(:,1:6),'rows');

Ndiffdata = size(tmp,1);

if Ndiffdata~=Nrow
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  msgtxt = ['The following data have identical  receiver coordinates, ';
            'field component indices, source parameters and frequency:'];
  disp(msgtxt);
  disp(' ');
  for ii = 1:Ndiffdata
    indpl=find(indj==indj(indi(ii)));
    if length(indpl)>1
      disp(['   In Lines ' num2str(indpl')]);
    end
  end
  errind = errind+1;
end

% ---- checking for negative frequency ----

negind = find(inptdat(:,6)<0);
if any(negind)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  msgtxt = ['There are inputs with negative frequency.                           ';
            'Check the 6th element of the following lines in the input data file:'];
  disp(msgtxt);
  disp(' ');
  disp(['Lines ' num2str(negind')]);
  disp(' ');
  disp('Frequency must be always nonnegative.');
  disp(' ');
  errind = errind+1;
end

% ---- checking for wrong source indices ----

badsrci = find( (inptdat(:,5)>length(srcpar)) | (inptdat(:,5)<1) );
if any(badsrci)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  msgtxt = ['The source index is nonpositive or larger than the number of source parameter.   ';
            'Check the 5th element of the following lines in the input data file:             '];
  disp(msgtxt);
  disp(' ');
  disp(['In lines ' num2str(badsrci')]);
  disp(' ');
  errind = errind+1;
end

% ---- checking for wrong field component indices ----

dind = inptdat(:,4);
badind = find( (dind>6)|(dind<=0) );
if any(badind);
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  msgtxt = ['There are impossible component indices in the input data.           ';
            'These numbers can be only -1 or -2 or 1 or 2 or 3 or 4 or 5 or 6.   ';
            'Check the 4th element of the following lines in the input data file:'];
  disp(msgtxt);
  disp(' ');
  disp(['In lines ' num2str(badind')]);
  disp(' ');
  errind = errind+1;
end

if errind
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  error(['There are errors in recpar.dat.']);
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [xyd,dn,f,zr,dind,srcparreal,indsrt,indinvsrt] = dataprep(wordy,tmp,srcpar,hl,sl,al);

[indsrt,indinvsrt] = datasrt([tmp(:,1:3) abs(tmp(:,4)) tmp(:,5:6)]);
inptdat = tmp(indsrt,:);

if wordy >=2; disp('inside dataperp'); end;

f         = unique(inptdat(:,6)); Nf   = length(f);
srcind    = unique(inptdat(:,5)); Nsrc = length(srcind);
srcparreal= srcpar(srcind);
dind      = unique(inptdat(:,4)); Ncomp= length(dind);
zr        = unique(inptdat(:,3)); Nzr  = length(zr);
Nd        = size(inptdat,1);
findexold = find(f==inptdat(1,6));
sindexold = find(srcind==inptdat(1,5));
dindexold = find(dind==inptdat(1,4));
zindexold = find(zr==inptdat(1,3));

xyd = cell(Nf,Nsrc,Nzr,Ncomp);
xyr =[]; cmpxyzd=[]; dn=[];
for ii = 1:Nd
  f1 = inptdat(ii,6); findex = find(f==f1);
  s1 = inptdat(ii,5); sindex = find(srcind==s1);
  d1 = inptdat(ii,4); dindex = find(dind==d1);
  z1 = inptdat(ii,3); zindex = find(zr==z1);
  fchng = (findex ~= findexold);
  schng = (sindex ~= sindexold);
  dchng = (dindex ~= dindexold);
  zchng = (zindex ~= zindexold);

  if (fchng | schng | dchng | zchng)
    xyd{findexold,sindexold,zindexold,dindexold}=xyr;
    xyr = [];
  end
  xyr = [xyr;inptdat(ii,[1 2])];

  if (fchng | schng)
    xr2 = cmpxyzd(:,1);
    yr2 = cmpxyzd(:,2);
    zr2 = cmpxyzd(:,3);
    id2 = cmpxyzd(:,4);

    ssrrcc=srcpar{srcind(sindexold)} ;
    if ssrrcc(1)==1
      [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,1);
      e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
      if ssrrcc(2)==1; e(:,[2 3])=0; h(:,[1 3])=0; end;
      if ssrrcc(2)==2; e(:,[1 3])=0; h(:,[2 3])=0; end;
    else
      [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,ssrrcc);
      e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
    end
    eh = [e h];
    for jj = 1:size(cmpxyzd,1)
      dp = eh(jj,id2(jj));
      dn = [dn;dp];
    end
    cmpxyzd = [];
  end

  cmpxyzd = [cmpxyzd;inptdat(ii,[1 2 3 4])];
  findexold = findex; sindexold = sindex; dindexold = dindex; zindexold = zindex;
end

xyd{findexold,sindexold,zindexold,dindexold}=xyr;

xr2 = cmpxyzd(:,1);
yr2 = cmpxyzd(:,2);
zr2 = cmpxyzd(:,3);
id2 = cmpxyzd(:,4);

%pause
%save tmpall

ssrrcc=srcpar{srcind(sindexold)};
if ssrrcc(1)==1
  [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,1);
  e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
  if ssrrcc(2)==1; e(:,[2 3])=0; h(:,[1 3])=0; end;
  if ssrrcc(2)==2; e(:,[1 3])=0; h(:,[2 3])=0; end;
else
  [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,ssrrcc);
  e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
end
eh = [e h];
for jj = 1:size(cmpxyzd,1)
  dp = eh(jj,id2(jj));
  dn = [dn;dp];
end

%save tmpall2
%pause


% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [xyd,dn,f,zr,dind,srcparreal,indsrt,indinvsrt] = dataprep_subtx(wordy, tmp,srcpar,hl,sl,al,subtx);
% DATAPREP_SUBTX
% origianl : DATAPREP
% preparing basic calculation configuration with SUBTX modification
%
% Ja:
% DATAPREP with subtx mod. Using variable subtx, assume that multiple Tx as one virtual Tx
% [2004-12-01] only, solflag <= 6 is acceptable
% Basically, when N by user srcpar, make gropu with  subtx.
% Therefire, # of final virtual Tx = N / subtx.
% Normal mode, subtx = 0 or 1 is equivalent
%

% datasrt : very important index sorting.
%
%   sorteddata = inptdata(indsrt,:)
%
% becomes organized in the following order:
%
% 1. fastest variable - receiver x coordinate
% 2. fastest variable - receiver y coordinate
% 3. fastest variable - receiver z coordinate
% 4. fastest variable - observed field component index
% 5. fastest variable - source index
% 6. fastest variable - frequency
%
% To recover the order of the original input one may apply


%   inptdata = sorteddata(indinvsrt,:)
[indsrt,indinvsrt] = datasrt([tmp(:,1:3) abs(tmp(:,4)) tmp(:,5:6)]);
inptdat = tmp(indsrt,:); % sorted

save tmpimp ;

%disp('inside dataperp');

f         = unique(inptdat(:,6)); Nf   = length(f);
srcind    = unique(inptdat(:,5));


% sub transmitter [2004-11-30]
Nsrc = length(srcind);
%Nsrc = length(srcind) / subtx
%pause

srcparreal= srcpar(srcind);
dind      = unique(inptdat(:,4)); Ncomp= length(dind);
zr        = unique(inptdat(:,3)); Nzr  = length(zr);
Nd        = size(inptdat,1);
findexold = find(f==inptdat(1,6));
sindexold = find(srcind==inptdat(1,5));
%sindexold_subtx = sindexold ;
dindexold = find(dind==inptdat(1,4));
zindexold = find(zr==inptdat(1,3));

xyd = cell(Nf,Nsrc,Nzr,Ncomp);
xyr =[]; cmpxyzd=[]; dn=[];
%pause
% loop for all lines of recapr.dat, so that very SLOQW
for ii = 1:Nd
  f1 = inptdat(ii,6); findex = find(f==f1);

  % src index check. complicated..[2004-12-01]
  s1 = inptdat(ii,5) ;
  sindex = find(srcind==s1) ;

  % [2004-12-08]
  % tmp��take off sub index modification

  %if subtx == 1 ;
    % subtx ==1 is as usual
  %  sindex_subtx = sindex ;
  %elseif subtx ~= 1;
  %  % subtx is not 1, sindex_subtx calculation is as follows
  %  if s1 <= subtx;
  %    sindex_subtx = 1;
  %  else ;
  %    srem = fix( (s1-1) / subtx);
  %    sindex_subtx = srem + 1 ;
  %  end
  %end

  d1 = inptdat(ii,4); dindex = find(dind==d1);
  z1 = inptdat(ii,3); zindex = find(zr==z1);
  fchng = (findex ~= findexold) ;
  schng = (sindex ~= sindexold) ;
  %schng_subtx = (sindex_subtx ~= sindexold_subtx)
  dchng = (dindex ~= dindexold) ;
  zchng = (zindex ~= zindexold) ;
  kxyd = 1;
  %if (fchng | schng_subtx | dchng | zchng)
  if (fchng | schng | dchng | zchng) ;
    xyd{findexold,sindexold,zindexold,dindexold}=xyr ;
    disp('update xyd')
    xyr = [];
  end
  %if (fchng | schng_subtx | dchng | zchng)%rem(s1,subtx) == 1;
    xyr = [xyr;inptdat(ii,[1 2])]
  %  disp('insert xyr')
 %   pause;
  %end
  if (fchng | schng)
    xr2 = cmpxyzd(:,1);
    yr2 = cmpxyzd(:,2);
    zr2 = cmpxyzd(:,3);
    id2 = cmpxyzd(:,4);

    ssrrcc=srcpar{srcind(sindexold)} ;
    if ssrrcc(1)==1
      [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,1);
      e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
      if ssrrcc(2)==1; e(:,[2 3])=0; h(:,[1 3])=0; end;
      if ssrrcc(2)==2; e(:,[1 3])=0; h(:,[2 3])=0; end;
    else
      [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,ssrrcc);
      e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
    end
    eh = [e h];
    for jj = 1:size(cmpxyzd,1)
      dp = eh(jj,id2(jj));
      dn = [dn;dp];
    end
    cmpxyzd = [];
  end

  cmpxyzd = [cmpxyzd;inptdat(ii,[1 2 3 4])];
  findexold = findex; sindexold = sindex; dindexold = dindex; zindexold = zindex;
  %sindexold_subtx = sindex_subtx ;
end

xyd{findexold,sindexold,zindexold,dindexold}=xyr;
findexold;
zindexold;
dindexold;

%xyd{findexold,sindexold_subtx,zindexold,dindexold}=xyr;

xr2 = cmpxyzd(:,1);
yr2 = cmpxyzd(:,2);
zr2 = cmpxyzd(:,3);
id2 = cmpxyzd(:,4);

ssrrcc=srcpar{srcind(sindexold)};
if ssrrcc(1)==1
  [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,1);
  e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
  if ssrrcc(2)==1; e(:,[2 3])=0; h(:,[1 3])=0; end;
  if ssrrcc(2)==2; e(:,[1 3])=0; h(:,[2 3])=0; end;
else
  [e,h]=green3d(f(findexold),hl,sl,al,xr2,yr2,zr2,ssrrcc);
  e = reshape(e,length(xr2),3);h = reshape(h,length(xr2),3);
end
eh = [e h];
for jj = 1:size(cmpxyzd,1)
  dp = eh(jj,id2(jj));
  dn = [dn;dp];
end

% end of without subtx
% begining of with subtx

% [2004-11-29]
% add sub-tx option
% for subtx, here simply sum up background electric field

%save dnbefore dn ;

if subtx ~= 1;

  num.txset = length(srcpar) / subtx ;
  num.rec   = Nd / Nf / num.txset;
  num.insrc = num.rec / subtx ;

  tmp = zeros(num.insrc , num.txset);
  tmp2 = [];
  tmp3 = [];

  % already we have dn as vector, then sum up for Tx
  fprintf('size of dn before sum up%g\n',length(dn));
  for kk = 1:Nf ;
    for ii = 1:num.txset ;
      % before moving next subset��clear and initialize variable tmp
      clear tmp;  tmp = zeros(num.insrc , num.txset);
      for jj = 1 : subtx
        fprintf('%g\n',(kk-1) * num.rec + (ii-1) * num.rec + num.insrc * (jj-1) + 1);
        fprintf('%g\n',(kk-1) * num.rec + (ii-1) * num.rec + num.insrc * jj);
        tmp(:,ii) = tmp(:,ii) + dn( (kk-1) * num.rec + (ii-1)*num.rec + num.insrc * (jj-1) + 1 : (kk-1) * num.rec + (ii-1)*num.rec + num.insrc * jj ) ;
      end; % end of one (each) subset
      % repeat number of subset, therefore store as ii-th subset
    end; % end of entire subset
    tmp2(:,kk) = tmp(:)
    % repeat this for entire frequency, now, vetorize this current frequency's entire subset
  end;
  clear dn ;

  dn = tmp2(:);
  %  save tmp tmp
  disp('end of dataprep_subtx')
  fprintf('size of dn after sum up%g\n',length(dn));

  % next, think about xyd
  % at this moment xyd has info. from all Txs. then squeeze only specified subtx group and make new xyd

  xydold = xyd ;
  clear xyd ;
  Nsrc = num.txset ;
  xyd = cell(Nf,Nsrc,Nzr,Ncomp) ;
  for ii = 1 : Nf ;
    for kk = 1 : Nzr
      for ll = 1: Ncomp
        for jj = 1 : Nsrc
          xyd(ii,jj,kk,ll) = xydold(ii,(jj-1)*subtx+1,kk,ll);
        end
      end
    end
  end
end

clear tmp tmp2 tmp3 xydold ;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [indsrt,indinvsrt] = datasrt(A0);
% [indsrt,indinvsrt] = datasrt(A0);
%
% Computes sorting and unsorting indices for the
% input data such that
%
%   sorteddata = inptdata(indsrt,:)
%
% becomes organized in the following order:
%
% 1. fastest variable - receiver x coordinate
% 2. fastest variable - receiver y coordinate
% 3. fastest variable - receiver z coordinate
% 4. fastest variable - observed field component index
% 5. fastest variable - source index
% 6. fastest variable - frequency
%
% To recover the order of the original input one may apply
%
%   inptdata = sorteddata(indinvsrt,:)
%
% A0 is an (Nd,6) matrix formed by the first six columns of the input
% data file inptdata.dat

M = size(A0,1);
N = size(A0,2);
[tmp,ii,jj] = unique(A0,'rows');
if length(ii)~=length(jj)
  error('There are repetitions in the data');
end

A0 = [(1:M)' A0];

[tmp1,ii1] = sort(A0(:,end));
A = A0(ii1,:);

iu1=0;
tmptmp1=unique(tmp1);
for aa1 = 1:length(tmptmp1)
  Nelem1 = length(find(tmp1==tmptmp1(aa1)));
  i1 = (iu1+1):(iu1+Nelem1);
  A1 = A(i1,:);
  [tmp2,ii2]= sort(A1(:,end-1));
  A(i1,:)= A1(ii2,:);

  iu2 = iu1;
  tmptmp2 = unique(tmp2);
  for aa2 = 1:length(tmptmp2)
    Nelem2 = length(find(tmp2==tmptmp2(aa2)));
    i2 = (iu2+1):(iu2+Nelem2);
    A2 = A(i2,:);
    [tmp3,ii3]= sort(A2(:,end-2));
    A(i2,:)= A2(ii3,:);

    iu3 = iu2;
    tmptmp3 = unique(tmp3);
    for aa3 = 1:length(tmptmp3)
      Nelem3 = length(find(tmp3==tmptmp3(aa3)));
      i3 = (iu3+1):(iu3+Nelem3);
      A3 = A(i3,:);
      [tmp4,ii4]= sort(A3(:,end-3));
      A(i3,:)= A3(ii4,:);

      iu4 = iu3;
      tmptmp4 = unique(tmp4);
      for aa4 = 1:length(tmptmp4)
       	Nelem4 = length(find(tmp4==tmptmp4(aa4)));
        i4 = (iu4+1):(iu4+Nelem4);
        A4 = A(i4,:);
        [tmp5,ii5]= sort(A4(:,end-4));
        A(i4,:)= A4(ii5,:);

        iu5 = iu4;
        tmptmp5 = unique(tmp5);
        for aa5 = 1:length(tmptmp5)
          Nelem5 = length(find(tmp5==tmptmp5(aa5)));
          i5 = (iu5+1):(iu5+Nelem5);
          A5 = A(i5,:);
          [tmp6,ii6]= sort(A5(:,end-5));
          A(i5,:)= A5(ii6,:);

          iu5=iu5+Nelem5;
        end

        iu4=iu4+Nelem4;
      end

      iu3=iu3+Nelem3;
    end

    iu2=iu2+Nelem2;
  end

  iu1=iu1+Nelem1;
end

indsrt = A(:,1);
[tmp,indinvsrt]=sort(indsrt);

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [dind,AF,XY,YX,XYYX]=dindchk(dind)

if length(setdiff(dind,[-2 -1 1 2 3 4 5 6]))
 error('There are impossible values of dind');
end

if (any(dind(:)>0)&any(dind(:)<0))
 error('Anomalous field OR magnetotellurics!');
end

XY = any(dind(:)==-1);
YX = any(dind(:)==-2);
XYYX=XY&YX;
AF=any(dind(:)>=1);

[b,ii,jj]=unique(dind);
if length(ii)~=length(jj)
  warning(['There are repetitions in dind. One component is taken ' ...
           'into account once. Also, dind is being sorted.']);
else

  if AF & (sort(dind(:))-dind(:))'*(sort(dind(:))-dind(:))
    warning('The elements of dind are not in order. It is being sorted.');
  elseif XYYX & (dind(1)~=-1 | dind(2)~=-2)
    warning('The elements of dind are not in order. It is being sorted.');
  end
end

if AF
 dind=b;
end
if XYYX
 dind=[b(2) b(1)];
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function disphelp;
txt = [...
'                                                                                  ';
' INTEM3DQL 3-D Electromagnetic Forward Modeling Based on IE methods               ';
' The code requires two data files (sigbody.dat and recpar.dat).                   ';
' Also, 12 parameters must be defined, either in the input file                    ';
' named intem3d.par or as command line input arguments.                            ';
' The command line calling syntax of the program is the following:                 ';
'                                                                                  ';
'    intem3dql par1=value1 par2=value2 ...                                         ';
'                                                                                  ';
' If intem3d is called without any input argument, all parameters                  ';
' must present in the input file. If a variable is defined both in                 ';
' the input file and as a command line argument, the command line                  ';
' argument value prevails. For example, typing                                     ';
'                                                                                  ';
'    intem3dql stg=1                                                               ';
'                                                                                  ';
' we execute the first stage only, no matter what is assigned to                   ';
' it in the input file.                                                            ';
'                                                                                  ';
' The following parameters can be specified as command line arguments:             ';
'                                                                                  ';
'     stg  --  List of stages to execute                                           ';
'   wordy  --  Messages: 0-silence, 1-basic messages, 2-all messages               ';
'    mfit  --  accuracy level for iterative full IE solution                       ';
'srcpar{i} --  parameters of source #i                                             ';
'    sig0  --  vector of background sigma                                          ';
'     hh0  --  vector of background layer thicknesses                              ';
'     an0  --  vector of background layer anisotropies                             ';
'       x  --  Cell center x-coordinates of the anomalous area (m)                 ';
'       y  --  Cell center y-coordinates of the anomalous area (m)                 ';
'       z  --  Cell center z-coordinates of the anomalous area (m)                 ';
'      dz  --  Vertical cell sizes (m)                                             ';
' solflag  --  Solver flag.                                                        ';
'   combo  --  number of cells to combine (only for solflag=7)                     ';
'                                                                                  '];

disp(txt);

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------
function displayfwdmfit(it,err,time)

disp(['It = '		num2str(it,'%1d')  repmat(' ',1,abs(6-length(num2str(it)))) ...
      'Relative misfit = ' num2str(err,'%6.3e') ...
      '  time = '    num2str(time,'%6.3e')]);

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function y = fftgab(x,sign)

%   FFTGAB(X,+1) is the same as IFFTN(X).
%   FFTGAB(X,-1) is the same as  FFTN(X).

%   The input array X may have any dimensionality.  If X
%   is multi-dimensional, then a true multi-dimensional DFT
%   will be computed.

  if (sign == -1)
    y = fftn(x);
  elseif (sign == 1)
    y = ifftn(x);
  else
   error('The second argument must be +1 or -1.');
  end

% ----------------------------------------------------------------------------

% ----------------------------------------------------------------------------

function vb = fillgb1(wordy,f,hh,sig,an,x,y,z,dz,N)
% vb = fillgb1(wordy,f,hh,sig,an,x,y,z,dz,N);
%
% precomputing electric green's tensor kernels inside a
% rectangular body discretized by a horizontally homogeneous mesh.
%
% vb   : struct array with Green's tensor kernels
%        vb.g     - (Nz,3,3,Nz) cell array, each cell is a
%                   (2*Nx-1,2*Ny-1) complex array of Green's
%                   tensors inside the body
%        vb.Nx    - Number of cells in x direction
%        vb.Ny    - Number of cells in y direction
%        vb.avg   - constants for truncated calculation
%        vb.indxt - x indices of untruncated kernel elements
%        vb.indyt - y indices of untruncated kernel elements
%
%        Modifications in the code can be easily done using
%        the struct array.
%
% wordy: flag for displaying the status of computation
% f    : frequency
% hh   : vector of layer thicknesses [] in case of hom.halfspace
% sig  : vector of layer conductivities (can be complex)
% an   : vector of layer anisotropies
% x    : vector of x coordinates of the cell centers
% y    : vector of y coordinates of the cell centers
% z    : vector of z coordinates of the cell centers
% dz   : vector of cell sizes in vertical direction
% N    : Number of cells in horizontal direction to consider

Nx=length(x); Ny=length(y); Nz=length(z); Nxyz=Nx*Ny*Nz;
Nx2_1=2*Nx-1;
Ny2_1=2*Ny-1;
N2x=pow2(ceil(log2(Nx2_1)));
N2y=pow2(ceil(log2(Ny2_1)));

if(Nx==1);dx=dz(1);else;dx=x(2)-x(1);end
if(Ny==1);dy=dz(1);else;dy=y(2)-y(1);end

Nxt=min([N;Nx]);
Nyt=min([N;Ny]);
indxt=Nx-Nxt+(1:2*Nxt-1);
indyt=Ny-Nyt+(1:2*Nyt-1);

inxall=1:Nx2_1;
inyall=1:Ny2_1;

[ixin,iyin]=ndgrid(indxt,indyt);
indin=sub2ind([Nx2_1 Ny2_1],ixin(:),iyin(:));
indout=setdiff(1:Nx2_1*Ny2_1,indin);
numout=length(indout);

xr=dx*(0:Nx-1);yr=dy*(0:Ny-1);zr=z;
[xm,ym,zm]=ndgrid(xr,yr,zr);

if numout
  inx=[0:Nx-1]+Nx;
  iny=[0:Ny-1]+Ny;
  pones=zeros(N2x,N2y);
  pones(1:Nx,1:Ny)=1;
  noutall=zeros(N2x,N2y);
  nout=ones(Nx2_1,Ny2_1);
  nout(indxt,indyt)=0;
  noutall(1:Nx2_1,1:Ny2_1)=nout;
  sumout=fftgab(fftgab(noutall,-1).*fftgab(pones,-1),1);
  tmp2=sumout(inx,iny);
end

vb.g=cell(Nz,3,3,Nz);
vb.avg=cell(Nz,3,3,Nz);
vfft=zeros(N2x,N2y);
for iz=1:Nz
 e   = green3d(f,hh,sig,an,xm,ym,zm,[-2 dx dy dz(iz) 0 0 z(iz)]);
 e   = reshape(e,Nx,Ny,Nz,3,3);
 tmp = q2full0(1,e);
 for ii=1:Nz
  for jj=1:3
   for kk=1:3
    valall=tmp(:,:,ii,jj,kk);
    valin=valall(indxt,indyt);
    vb.g{ii,jj,kk,iz}=valin;
    if numout
      vfft(1:Nx2_1,1:Ny2_1)=valall;
      vfft(indxt,indyt)=0;
      sumout1=fftgab(fftgab(vfft,-1).*fftgab(pones,-1),1);
      tmp1=sumout1(inx,iny);
      avg=((tmp2(:))'*tmp1(:))/((tmp2(:))'*tmp2(:));
      vb.avg{ii,jj,kk,iz}=avg;
    else
      vb.avg{ii,jj,kk,iz}=0;
    end

    % precompute FFT(vb.g) [2005-12-19]
    vfft=zeros(N2x,N2y);
    vfft(inxall,inyall)=vb.avg{ii,jj,kk,iz} ;
    vfft(indxt,indyt)=valall ;

    % applying FFT
    vb.gfft{ii,jj,kk,iz} = fftgab(vfft,-1) ;

   end
  end
 end
end
vb.Nx=Nx;
vb.Ny=Ny;
vb.indxt=indxt;
vb.indyt=indyt;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function vr = fillgr1(wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody)
% vr = fillgr1(wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody)
%
% precomputing body-receiver green's tensor kernels
% for a rectangular body discretized by a horizontally
% homogeneous mesh. The horizontal locations of receivers
% coincide the cell centers.
%
% vr   : struct array with Green's tensor kernels
%        vr.g     - (Nz,3,3,Nz) cell array, each cell is a
%                   (2*Nx-1,2*Ny-1) complex array of Green's
%                   tensors inside the body
%        vr.Nx    - Number of cells in x direction
%        vr.Ny    - Number of cells in y direction
%        vr.avg  - constants for truncated calculation
%        vr.indxt - x index for FFT multiplication
%        vr.indyt - y index for FFT multiplication
%        vr.indbody - indices of the true body in the auxiliary body
%                     incorporating receivers
%
%        Modifications in the code can be easily done using
%        the struct array.
%
% wordy: flag for displaying the status of computation
% f    : frequency
% hh   : vector of layer thicknesses [] in case of hom.halfspace
% sig  : vector of layer conductivities (can be complex)
% an   : vector of layer anisotropies
% x    : vector of x coordinates of the cell centers
% y    : vector of y coordinates of the cell centers
% z    : vector of z coordinates of the cell centers
% dz   : vector of cell sizes in vertical direction
% N    : Number of cells in horizontal direction to consider
% zr   : vector of z coordinates of the receivers
% dind : vector describing what data is measured.
%         1 -> Ex;  2 -> Ey;  3 -> Ez;
%         4 -> Hx;  5 -> Hy;  6 -> Hz;
%        -1 -> MT xy mode; -2 -> MT yx mode.
%         For example dind = [1 3 4 5] means that the Green's
%         matrix corresponding to Ex, Ez, Hx and Hy will return.
%         MT and CS fields cannot be computed togeter.
% indbody - indices of the true body in the auxiliary body
%             incorporating receivers


[dind,AF,XY,YX,XYYX]=dindchk(dind);

Nx=length(x); Ny=length(y); Nz=length(z);
Nxyz=Nx*Ny*Nz;Nzr=length(zr);
Nx2_1=2*Nx-1;
Ny2_1=2*Ny-1;
N2x=pow2(ceil(log2(Nx2_1)));
N2y=pow2(ceil(log2(Ny2_1)));
Nxt=min([N;Nx]);
Nyt=min([N;Ny]);
Ncomp = length(dind);

if(Nx==1);dx=dz(1);else;dx=x(2)-x(1);end
if(Ny==1);dy=dz(1);else;dy=y(2)-y(1);end

indxt=Nx-Nxt+(1:2*Nxt-1);
indyt=Ny-Nyt+(1:2*Nyt-1);

inxall=1:Nx2_1;
inyall=1:Ny2_1;

[ixin,iyin]=ndgrid(indxt,indyt);
indin=sub2ind([Nx2_1 Ny2_1],ixin(:),iyin(:));
indout=setdiff(1:Nx2_1*Ny2_1,indin);
numout=length(indout);
inde = dind(find(dind<=3));
indh = dind(find(dind>3))-3;

xr=dx*(0:Nx-1);yr=dy*(0:Ny-1);
[xm,ym,zm]=ndgrid(xr,yr,zr);


if numout
  inx=[0:Nx-1]+Nx;
  iny=[0:Ny-1]+Ny;
  pones=zeros(N2x,N2y);
  pones(1:Nx,1:Ny)=1;
  noutall=zeros(N2x,N2y);
  nout=ones(Nx2_1,Ny2_1);
  nout(indxt,indyt)=0;
  noutall(1:Nx2_1,1:Ny2_1)=nout;
  sumout=fftgab(fftgab(noutall,-1).*fftgab(pones,-1),1);
  tmp2=sumout(inx,iny);
end

vr.g=cell(Nzr,Ncomp,3,Nz);
vr.avg=cell(Nzr,Ncomp,3,Nz);
vfft=zeros(N2x,N2y);

[e0,h0] = green3d(f,hh,sig,an,zeros(size(zr)),zeros(size(zr)),zr,1);
e0=reshape(e0,length(zr),3);
h0=reshape(h0,length(zr),3);

tmp=zeros(Nx2_1,Ny2_1,Nzr,Ncomp,3);
Ny2_1=2*Ny-1;

for iz=1:Nz
 [e,h]=green3d(f,hh,sig,an,xm,ym,zm,[-1 dx dy dz(iz) 0 0 z(iz)]);
 e = reshape(e,Nx,Ny,Nzr,3,3);h = reshape(h,Nx,Ny,Nzr,3,3);
 tmpe = q2full0(1,e);tmph = q2full0(2,h);

 if   AF; tmp=cat(4,tmpe(:,:,:,inde,:),tmph(:,:,:,indh,:)); end;
 for ii=1:Nzr

  if XY
    tmpte=2*(tmpe(:,:,ii,1,:)/e0(ii,1)-tmph(:,:,ii,2,:)/h0(ii,2));
    tmpte(:,:,:,:,[2 3])=0;  % !!! For full forward problem it must be disabled !!!
    tmp(:,:,ii,1,:)=tmpte;
  end
  if YX
    tmpth=2*(tmpe(:,:,ii,2,:)/e0(ii,2)-tmph(:,:,ii,1,:)/h0(ii,1));
    tmpth(:,:,:,:,[1 3])=0;  % !!! For full forward problem it must be disabled !!!
    tmp(:,:,ii,1,:)=tmpth;
  end
  if XYYX; tmp(:,:,ii,:,:)=cat(4,tmpte,tmpth); end;

  for jj=1:Ncomp
   for kk=1:3
    valall=tmp(:,:,ii,jj,kk);

    if (any(valall(:)))
      valin=valall(indxt,indyt);
      vr.g{ii,jj,kk,iz}=valin;
      if numout
        vfft(1:Nx2_1,1:Ny2_1)=valall;
        vfft(indxt,indyt)=0;
        sumout1=fftgab(fftgab(vfft,-1).*fftgab(pones,-1),1);
        tmp1=sumout1(inx,iny);
        avg=((tmp2(:))'*tmp1(:))/((tmp2(:))'*tmp2(:));
        vr.avg{ii,jj,kk,iz}=avg;
      else
        vr.avg{ii,jj,kk,iz}=0;
      end
    else
      vr.g{ii,jj,kk,iz}=[];
      vr.avg{ii,jj,kk,iz}=[];
    end

    % precompute FFT(vr.g) [2005-12-19]
    vfft=zeros(N2x,N2y);
    %save tmpall
    %vfft(inxall,inyall)=vr.avg{ii,jj,kk,iz} ;
    vfft(indxt,indyt)=valall ;
    % applying FFT
    vr.gfft{ii,jj,kk,iz} = fftgab(vfft,-1) ;

   end
  end
 end
end
vr.Nx=Nx;
vr.Ny=Ny;
vr.indxt=indxt;
vr.indyt=indyt;
vr.indbody=indbody;
vr.Nxyz=Nxyz;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function wi = fillwi(x,y,xyd);
% wi = fillwi(x,y,xyd);
%
% filling interpolation matrices for each frequency,
% source, receiver z-level and field component.
%
% wi  - struct array of interpolation matrices, irregular
%       data indices, data and body parameters.
%       wi.w    - Nf,Nsrc,Nzr,Ncomp cell array of
%                  interpolation matrices
%       wi.indl - Nf,Nsrc,Nzr,Ncomp double array of
%                  lower indices in the irregular data vector
%       wi.indu - Nf,Nsrc,Nzr,Ncomp double array of
%                  upper indices in the irregular data vector
%       wi.Nf   - number of frequencies
%       wi.Nsrc - number of sources
%       wi.Nzr  - number of receiver z-levels
%       wi.Ncomp - number of observed field components
%
% x   - vector of x coordinates of cell centers of the
%        anomalous body
%
% y   - vector of y coordinates of cell centers of the
%        anomalous body
%
% xyd -  Nf,Nsrc,Nzr,Ncomp cell array of xy coordinates of
%        the irregular data for each freq, src,
%        rec. z-level and field component


Nf = size(xyd,1);
Nsrc = size(xyd,2);
Nzr = size(xyd,3);
Ncomp =  size(xyd,4);
%xyd
%size(xyd)
%whos xyd
%pause
cntu = 0;
wd.w=cell(Nf,Nsrc,Nzr,Ncomp);
wd.indl=zeros(Nf,Nsrc,Nzr,Ncomp);
wd.indu=zeros(Nf,Nsrc,Nzr,Ncomp);

for ii = 1:Nf
  for jj = 1:Nsrc
    for ll = 1:Ncomp
      for kk = 1:Nzr
        xytmp = xyd{ii,jj,kk,ll};
%        ii
%        jj
%        ll
%        kk
%        pause;
        if size(xytmp,1)>0
%        disp('in in') ;
%        pause;
          wi.w{ii,jj,kk,ll}=blin2dm(x,y,xytmp);
          cntl=cntu+1;
          cntu=cntu+size(xytmp,1);
          wi.indl(ii,jj,kk,ll)=cntl;
          wi.indu(ii,jj,kk,ll)=cntu;

        end

      end
    end
  end
end

wi.Nf = Nf;
wi.Nsrc = Nsrc;
wi.Nzr = Nzr;
wi.Ncomp = Ncomp;
wi.Nxy = length(x)*length(y);

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function b = fwdrec(wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody,m,et,wi);
% b = fwdrec(wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody,m,et,wi);
%
% Calculates the anomalous fields at the receivers
%
% wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody : see fillgr1.m
% b    : Nd,1 vector of anomalous fields at the receivers
%        Nd is the number of data
%
% m    : Nxyz,1 anomalous conductivity vector, Nxyz is the
%         number of model parameters (cells)
%
% et   : struct array of total electric fields inside the anomalous body
%
% wi   : struct array of interpolation matrices, irregular
%        data indices, data and body parameters.
%         (see fillwi.m for details)

Nf = wi.Nf;
Nsrc = wi.Nsrc;
Nzr = wi.Nzr;
Ncomp = wi.Ncomp;
Nxy = wi.Nxy;
Nd=max(wi.indu(:));

b=zeros(Nd,1);
for ii = 1:Nf
  vr = fillgr1(wordy,f(ii),hh,sig,an,x,y,z,dz,N,zr,dind,indbody);
  for jj = 1:Nsrc
    breg = reshape(multa1(m,et.e{ii,jj},vr),Nxy,Nzr,Ncomp);
    for kk = 1:Nzr
      for ll = 1:Ncomp
        if nnz(wi.w{ii,jj,kk,ll})
          b(wi.indl(ii,jj,kk,ll):wi.indu(ii,jj,kk,ll)) = wi.w{ii,jj,kk,ll}*breg(:,kk,ll);
        end
      end
    end
  end
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function res=getpar(pars,kind,name,def,cmnt)
% Grabbing the parameter
% from the input buffer
%
%

if(length(pars)==0)
   disp(sprintf('%8s = %-12s  %s',name,def,cmnt))
   res=[];
else
   in=min(find(strcmp(pars(:,1),name)));
   res=char(pars(in,2));
end

if(isempty(res)) res=def; end;
if(isempty(res))  return; end;

switch(lower(kind))
  case 'char'
    if(ischar(res)==0) res=num2str(res); end;
  case 'double'
    if(ischar(res))

       % colon conversion here
       % vectors without colons are converted anyway by str2num
       %

       in=find(res==':');
       inb=find( (res=='[')+(res==']') );

       switch(length(in))
        case 0
         res=str2num(res);
         return;
        case 1
         res(inb)=' ';
         stp=1;
         beg=str2num(res(1:in-1));
         fin=str2num(res(in+1:end));
        case 2
         res(inb)=' ';
         beg=str2num(res(1:in(1)-1));
         stp=str2num(res((in(1)+1):(in(2)-1)));
         fin=str2num(res((in(2)+1):end));
        otherwise
        disp(['Error in parameter ' name ' bad value ' res ]);
       end
       res=[ beg : stp : fin ];

    end
end;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function x = gmrgab(wordy,vb,ds3,b,m1,m2,x0,phi0,s);
%
% x = gmrgab(wordy,vb,ds3,b,m1,m2,x0,phi0,s);
%
% Generalized minimum residual for forward problem
% 6/23/2004 bug fix, and exit in a internal step by Ken Yoahioka
%
% wordy - if wordy>1:display errors; if wordy<1: no messages
% A     - coefficient matrix
% b     - right hand side
% m1    - left preconditioner matrix
% m2    - right preconditioner matrix
% x0    - initial guess
% phi0  - stopping error (relative residual)
% s     - number of inner iterations

Nm = length(ds3);
%Nm = size(A,1);
if length(m1)==0; m1=ones(Nm,1); end;
if length(m2)==0; m2=ones(Nm,1); end;

% ----- integral equation stuff

m1m2 = m1.*m2;
sam2 = ds3.*m2;
b  = m1.*b;
x  = x0./m2;
r  = x.*m1m2 - m1.*multgb1(vb,sam2.*x) - b; % r=A*x-b

% -------------------------------------

mf0 = b'*b;
  g = zeros(Nm,s);
 Ag = zeros(Nm,s);
  k = zeros(s,1);
Ag2 = zeros(s,1);
tic;

iit = 0; flag =1;
it=1;
while (it <= Nm*2 & flag)
%%for it=1:Nm*2
  iit = iit +1;
  rn  = x.*m1m2 - m1.*multgb1(vb,sam2.*x) - b; % r=A*x-b

  mfit = abs(rn'*rn);
  err  = sqrt(mfit/mf0);
  %%if(wordy>1); displayfwdmfit(it,err,toc); end;
  if(wordy>1); displayfwdmfit(iit,err,toc); end;
  if(err<phi0); flag =0; break; end;

  g(:,1)  = rn;
  Ag(:,1) = g(:,1).*m1m2 - m1.*multgb1(vb,sam2.*g(:,1)); % Ag=A*g
  Ag2(1)  = (Ag(:,1))'*(Ag(:,1));
  %% k(1)    = (rn'*Ag(:,1))/Ag2(1);
  k(1)    = (Ag(:,1)' * rn) / Ag2(1);
  x       = x - k(1)*g(:,1);

  rst = rn;
  for p = 2:s
    rst     = rst - k(p-1)*Ag(:,p-1);
    % START: Check RMS in a internal step
    iit = iit +1;
    mfit = abs(rst'*rst);
    err  = sqrt(mfit/mf0);
    if(wordy>1); displayfwdmfit(iit,err,toc); end;
    if(err<phi0); flag = 0; break; end;
    % END: Check RMS in a internal step
    Arst    = rst.*m1m2 - m1.*multgb1(vb,sam2.*rst);  % Arst = A*rst
    g(:,p)  = rst;
    for l = 1:p-1
      %% beta   = (Arst'*Ag(:,l))/Ag2(l);
      beta   = (Ag(:,l)' * Arst) / Ag2(l);
      g(:,p) = g(:,p) - beta*g(:,l);
    end
    Ag(:,p) = g(:,p).*m1m2 - m1.*multgb1(vb,sam2.*g(:,p)); % Ag=A*g
    Ag2(p)  = (Ag(:,p))'*(Ag(:,p));
    %% k(p)    = (rn'*Ag(:,p))/Ag2(p);
    k(p)    = (Ag(:,p)' * rn) / Ag2(p);
    x       = x - k(p)*g(:,p);
  end
end

x=x.*m2;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [x,y,z,dz]= inptchk(hl,sl,al,x,y,z,dz);

%   -------- code starts here -------

x = x(:); y = y(:); z = z(:);
dz = dz(:);
Nx=length(x); Ny=length(y); Nz=length(z);
%nl = length(sl);
nl = length(hl)+1;

errflg = 0;

if ( any(sl<=0)|any(al<=0)|any(hl<=0) )
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['There are nonpositive layer conductivities, anisotropies or thicknesses.']);
  disp(['Check out sl, al and hl in the input parameter file.	 ']);
  errflg = errflg+1;
end

if nl ~= length(al)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['The lengths of the layer conductivity and anisotropy vectors are']);
  disp(['different. Check out al and sl in the input parameter file.     ']);
  errflg = errflg+1;
end

if nl ~= length(hl)+1
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['The length of the layer conductivity vector is inconsistent with the ']);
  disp(['thickness vector. Check out hl and sl in the input parameter file.	 ']);
  errflg = errflg+1;
end

if (any( (diff(sl)==0) & (diff(al)==0) ))
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['There are neighboring layers with identical conductivities and ']);
  disp(['anisotropies. This may cause problems for Green''s tensor and normal']);
  disp(['field calculations. These layers must be united into one.']);
  disp(['Check out sl and al in the input parameter file.']);
  errflg = errflg+1;
end

if Nx < 2
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['There is ' num2str(Nx) ' cell in x direction. It must be at least 2.']);
  errflg = errflg+1;
end

if Ny < 2
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['There is ' num2str(Ny) ' cell in y direction. It must be at least 2.']);
  errflg = errflg+1;
end

% mod by Takumi Ueda on 2004-0624
%if ( any(std(diff(x))) | any(std(diff(y))) )
if (min(std(diff(x))) > 1e-8) | (min(std(diff(y))) > 1e-8)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['The cell centers of the inverted area must form a horizontally homogeneous grid.']);
  errflg = errflg+1;
end

if ( x(1) >= x(end) )
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['The cell center x coordinates have been sorted into an ascending order.']);
  x = flipud(x);
end

if ( y(1) >= y(end) )
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['The cell center y coordinates have been sorted into an ascending order.']);
  y = flipud(y);
end

if any(diff(z)<0)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['The cell center z coordinates have been sorted into an ascending order.']);
  [z,ind] = sort(z);
  dz = dz(ind);
end

if Nz ~= length(dz)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['z and dz must have the same length in the input parameter file.']);
  errflg = errflg+1;
end

if any(dz<=0)
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['Vertical cell sizes must be positive. ']);
  disp(['Check out dz in the input parameter file.']);
  errflg = errflg+1;
end

zmin = z(:)-dz(:)/2;
zmax = z(:)+dz(:)/2;

if any(zmin<0)
%if any(abs(zmin)>1e-16) % modified 2004-0820
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  disp(['There are cells above the surface. ']);
  disp(['Check out z and dz in the input parameter file.']);
  errflg = errflg+1;
end

hl=[0 hl inf];

zbnd=zeros(1,length(hl));
zbnd(1)=hl(1);
for i=2:length(hl)
   zbnd(i)=zbnd(i-1)+hl(i);
end


for ii = 1:Nz
  if any((zbnd<zmax(ii))&(zbnd>zmin(ii)))
    disp(' ');
    disp('-------------------------------------------------------------------------');
    disp(' ');
    disp(['There are layer boundaries intersecting cells. ']);
    disp(['Check out z, dz and hl in the input parameter file.']);
    errflg = errflg+1;
  end
end

for ii = 1:Nz
  if (( any((zmin+0.01<zmax(ii))&(zmin>zmin(ii)+0.01)) )|( any((zmax+0.01<zmax(ii))&(zmax>zmin(ii)+0.01)) ))
    disp(' ');
    disp('-------------------------------------------------------------------------');
    disp(' ');
    disp(['There are cells intersecting each other vertically. ']);
    disp(['Check out z and dz in the input parameter file.']);
    errflg = errflg+1;
    break;
  end
end


if errflg
  disp(' ');
  disp('-------------------------------------------------------------------------');
  disp(' ');
  error(['There are errors in the input parameter file.']);
end


% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function res=loadasc(from)
% loadasc - load in ascii format
%
%

in=fopen(from,'r');
line=fgetl(in);    % determine the length
res=sscanf(line,'%g');
d2=length(res);

fseek(in,0,-1);
res=fscanf(in,'%g');

fclose(in);

l=length(res);
d1=round(l/d2);
if(d1*d2==l) res=reshape(res,d2,d1)'; end

return;

function sigtot=loadascc(from,kcomp,f)
%
% modified loadsscc
% enable to compute complex resistivity and conductivity
%
% loadasc - load in ascii format
% loadascc - read in complex value (only for sigbody.dat)
% kcomp(2) 0 = real, 1 = complex, 2 = Cole-Cole
%
  switch kcomp(2)
   %
   % conductivity of 3D domain is defined by real number
   %
   case 0
    in=fopen(from,'r');
    line=fgetl(in);    % determine the length
    res=sscanf(line,'%g');
    d2=length(res);

    fseek(in,0,-1);
    res=fscanf(in,'%g');

    fclose(in);

    l=length(res);
    d1=round(l/d2);
    if(d1*d2==l) res=reshape(res,d2,d1)'; end
    if(d1*d2==l) sigtot=reshape(res,d2,d1)'; end
   %
   % conductivity of 3D domain is defined by real and imaginary number
   %
   case 1
    stcomp=load('sigbody.dat');
    if size(stcomp,2)~=2;error('Both of real and comples value are required in sigbody.dat');end;
%    res=stcomp(:,1)+1i*stcomp(:,2);
    sigtot=stcomp(:,1)+1i*stcomp(:,2);
   %
   % conductivity of 3D domain is defined by four Cole-Cole parameters
   %
   case 2;
    stcomp=load('sigbody.dat');
    if size(stcomp,2)~=4;error('All Cole-Cole parameters are required in sigbody.dat');end;
    sigr0=stcomp(:,1);
    chrgabl0=stcomp(:,2);
    timeconst0=stcomp(:,3);
    freqconst0=stcomp(:,4);
    %whos;
    [res_comp,sigma_comp]=ipp2csig(sigr0.^(-1),chrgabl0,timeconst0,freqconst0,f');


    if length(sigma_comp{1})==1
      for ii=1:length(f)
        sigtot{ii} = sigma_comp{ii}(1)*ones(Nxyz,1);
      end
    elseif (length(sigma_comp{1})==Nxyz)
      sigtot = sigma_comp;
    else
      error('The number of Cole-Cole parameters in [sigbody.dat] does not match [intem3d.par]');
    end
    res=sigma_comp(:,1);
  end
  return;


% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function g2=mirror(g1,dim)
% g2=mirror(g1,dim)
%
% MIRROR  - concatenates a matrix and its mirror image
%           into a larger matrix along dimension dim.
% g1 - input array
% dim - dimension along which we perform the reflection
%       for now dim=1 or dim=2 is allowed only.
%
% g2 - output array, its dim-th size will increase
%      from N to 2*N-1.
%

if (dim<=0); error('Negative dimension is not allowed'); end
if (dim>=3); error('Only the first and second dimension can be mirrored.'); end
if (ndims(g1))<dim ; error('Concatenation along nonexistent dimension is not allowed'); end

if dim==1; g2=cat(1,g1(end:-1:2,:,:,:,:),g1); end;
if dim==2; g2=cat(2,g1(:,end:-1:2,:,:,:),g1); end;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function [m1,m2]=mkprecnd(ds3,sb3);
% [m1,m2]=mkprecnd(ds3,sb3);
% producing left and right diagonal preconditioners
% based on the modified Green's operator
%
% ds3 - [ds,ds,ds] anomalous conductivity vector
% sb3 - [sb,sb,sb] background conductivity vector
%
% m1 - left preconditionter
% m2 - right preconditionter


m1 = sqrt(sb3);
m2 = (2*m1)./(ds3+2*sb3);

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function b = multa1(p,eb,vr);
% b = multa1(p,eb,vr);
%
% multiplication by matrix A.
%
% b    : Nx*Ny*Nzr*Ncomp output column vector
% p    : Nx*Ny*Nz input column vector
% eb   : 3*Nx*Ny*Nz,1 vector of
%        background electric field inside the body
%        (produced by filleb1)
% vr   : struct array with Green's tensor kernels
%        (see multgr1 for details)

Nxyz=vr.Nxyz;
indbody=vr.indbody;
ind3=[indbody;Nxyz+indbody;2*Nxyz+indbody];

pbig=zeros(Nxyz,1);
pbig(indbody)=p;
p3    = repmat(pbig,3,1);

ebbig=zeros(Nxyz*3,1);
ebbig(ind3)=eb(:);

b=multgr1(vr,ebbig.*p3);

%save mulc;
%disp('save muls')
%pause

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function b = multgb1(vb,p);
%  b = multgb1(vb,p);
%
%  Multiplication by matrix G inside the body using FFT exploiting
%  horizontally homogeneous discretization of the anomalous body.
%
%  FFT-covolution algorithm has been modified by T.Ueda based on
%  PIE3D(Yoshioka, 2004-) implementation.
%
%  b = G*p;
%
%  See also: MULTGBT, MULTGBCT
%
%  Gabor Hursan - 2000.
%  Takumi Ueda  - 2005 -
%
% b = multgb(vb,p);
%
% b  : 3*Nx*Ny*Nz output column vector
% vb   : struct array with Green's tensor kernels
%  vb.g - (Nz,3,3,Nz) cell array, each cell is a
%     (2*Nx-1,2*Ny-1) complex array of Green's
%     tensors inside the body
%  vb.gfft - (Nz,3,3,Nz) cell array, each cell is a
%     (N2x,N2y) complex array of FFT results of Green's
%     tensors inside the body
%  vb.Nx - Number of cells in x direction
%  vb.Ny - Number of cells in y direction
%  vb.avg - constants for truncated calculation
%  vb.indxt - x indices of untruncated kernel elements
%  vb.indyt - y indices of untruncated kernel elements
%
%  Modifications in the code can be easily done using
%  the struct array.
%
%  p : 3*Nx*Ny*Nz input column vector
%

if (size(p,2) ~= 1) error('The vector argument must be column'); end;

% ---- Numbers -----

Nz=size(vb.g,1);
Nx=vb.Nx;
Ny=vb.Ny;
Nxyz=Nx*Ny*Nz;
Nxy=Nx*Ny;
Nm=Nxyz*3;
Nx2_1=2*Nx-1;
Ny2_1=2*Ny-1;
N2x=pow2(ceil(log2(Nx2_1)));
N2y=pow2(ceil(log2(Ny2_1)));

% ---- Indices -----

inx=[0:Nx-1]+Nx;
iny=[0:Ny-1]+Ny;
inxall=1:Nx2_1;
inyall=1:Ny2_1;
indxt=vb.indxt;
indyt=vb.indyt;

% ---- reshaping the vector argument and        ----
% ---- allocating arrays for FFT-multiplication ----

p = reshape(p(:),Nx,Ny,Nz,3);
%vfft=zeros(N2x,N2y);
b=zeros(Nm,1);
ptmp=cell(3,Nz);

% ---- multiplication starts ----
for izc=1:Nz   % ---- source cells ----

   %b3  =zeros(Nm,1);
   %v   =vb.g(izc,:,:,:);
   % pre compute vector p as cell array for Ex, Ey, Ez ;

   for ii=1:3  % ---- alpha component ----

       b1 = zeros(N2x,N2y);

       % index for which set now we are computing
       ind= (1:Nxy) + (izc-1)*Nxy + (ii-1)*Nxyz ;

       %       indcell{(izc-1)*Nz+ii} = ind ;

       for izr=1:Nz   % ---- receiver cells ----

           b0 = zeros(N2x,N2y);

           if izc == 1 & ii== 1 ;
               pfft=zeros(N2x,N2y);
               for kk=1:3
                   pfft(1:Nx,1:Ny)=squeeze(p(:,:,izr,kk));
                   if any(pfft(:))
                       ptmp{kk,izr}=fftgab(pfft,-1);
                   else
                       ptmp{kk,izr}=pfft;
                   end
               end
           end

           for jj=1:3  % ---- beta component ----

               %vtmp=v{1,ii,jj,izr}; % OK

               pfft=ptmp{jj,izr}; % OK

               gfft = vb.gfft{izc,ii,jj,izr} ;

               if ( any(pfft(:)) & any(gfft(:)) )
                   %vfft(inxall,inyall)=vb.avg{izc,ii,jj,izr} ;
                   %vfft(indxt,indyt)=vtmp ;

                   % btmp=fftgab(vfft,-1).*pfft ;
                   % btmp=fftgab(fftgab(vfft,-1).*pfft,1);
                   btmp = gfft.*pfft ;

                   b0 = b0 + btmp; % sum up for Z-dir
               end

           end % end of loop for Nz

           b1 = b1 + b0 ; % sum up for beta (jj)

       end % end of loop for beta (jj)

       % now that we are done for dual-sum-up, then apply IFFT
       btmp2 = fftgab(b1,1) ;

       % squezze only we need (actual cell index)

       b2 = btmp2(inx,iny) ;

       % this is at izc-th layer and alpha=ii component
       b(ind) = b2(:) ;
   end

end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function b = multgbct1(vb,p);
%  b = multgbct1(vb,p);
%
%  Multiplication by the complex conjugate transpose of
%  matrix G inside the body using FFT exploiting
%  horizontally homogeneous discretization of the anomalous body.
%
%  b = G'*p;
%
%  See also: MULTGB, MULTGBT
%
%  Gabor Hursan - 2000.
%
% b = multgbct(vb,p);
%
% b  : 3*Nx*Ny*Nz output column vector
% vb   : struct array with Green's tensor kernels
%  vb.g - (Nz,3,3,Nz) cell array, each cell is a
%     (2*Nx-1,2*Ny-1) complex array of Green's
%     tensors inside the body
%  vb.Nx  - Number of cells in x direction
%  vb.Ny  - Number of cells in y direction
%  vb.avg - constants for truncated calculation
%  vb.indxt - x indices of untruncated kernel elements
%  vb.indyt - y indices of untruncated kernel elements
%
%  Modifications in the code can be easily done using
%  the struct array.
%
%  p : 3*Nx*Ny*Nz input column vector
%

if (size(p,2) ~= 1) error('The vector argument must be column'); end;
Nz=size(vb.g,1);
Nx=vb.Nx;
Ny=vb.Ny;
Nxyz=Nx*Ny*Nz;
Nxy=Nx*Ny;
Nm=Nxyz*3;
Nx2_1=2*Nx-1;
Ny2_1=2*Ny-1;
N2x=pow2(ceil(log2(Nx2_1)));
N2y=pow2(ceil(log2(Ny2_1)));

p = reshape(p(:),Nx,Ny,Nz,3);

inx=[0:Nx-1]+Nx;
iny=[0:Ny-1]+Ny;
inxall=1:Nx2_1;
inyall=1:Ny2_1;
indxt=vb.indxt;
indyt=vb.indyt;

vfft=zeros(N2x,N2y);
b=zeros(Nm,1);

pfft=zeros(N2x,N2y);
ptmp=cell(Nz,3);
for kk=1:Nz
   for ll=1:3
      pfft(1:Nx,1:Ny)=squeeze(p(:,:,kk,ll));
      if any(pfft(:))
         ptmp{kk,ll}=fftgab(pfft,-1);
      end
   end
end

for izc=1:Nz
   v = vb.g(:,:,:,izc);
   for izr=1:Nz
      for ii=1:3  % cell
         b1=zeros(Nx,Ny);
         ind=(1:Nxy)+(izc-1)*Nxy+(ii-1)*Nxyz;
         for jj=1:3   % receiver
           pfft=ptmp{izr,jj};
	   vtmp=conj(flipdim(flipdim(v{izr,jj,ii},1),2));
           if ( any(pfft(:)) & any(vtmp(:)) )
             vfft(inxall,inyall)=conj(vb.avg{izr,ii,jj,izc});
             vfft(indxt,indyt)=vtmp;
             btmp=fftgab(fftgab(vfft,-1).*pfft,1);
             b1=b1+btmp(inx,iny);
           end
         end
      b(ind)=b(ind)+b1(:);
      end
   end
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function b = multgr1(vr,p);
%  b = multgr1(vr,p);
%
%  Multiplication by matrix Gr (body-receiver) using FFT exploiting
%  horizontally homogeneous discretization of the anomalous body.
%
%  b = Gr*p;
%
%  See also: MULTGRT, MULTGRCT
%
%  Gabor Hursan - 2000.
%
% b = multgr(vr,p);
%
% b  : Nx*Ny*Nzr*Ncomp output column vector
% vr   : struct array with Green's tensor kernels
%  vr.g - (Nz,3,3,Nz) cell array, each cell is a
% (2*Nx-1,2*Ny-1) complex array of Green's
%      tensors inside the body
%  vr.Nx  - Number of cells in x direction
%  vr.Ny  - Number of cells in y direction
%  vr.avg - constants for truncated calculation
%  vr.indxt - x indices of untruncated kernel elements
% vr.indyt - y indices of untruncated kernel elements
%
%  Modifications in the code can be easily done using
%  the struct array.
%
%  p : 3*Nx*Ny*Nz input column vector
%

if (size(p,2) ~= 1) error('The vector argument must be column'); end;

% ---- Numbers -----

Nzr=size(vr.g,1);
Ncomp=size(vr.g,2);
Nz=size(vr.g,4);
Nx=vr.Nx;
Ny=vr.Ny;
Nxyz=Nx*Ny*Nz;
Nxy=Nx*Ny;
Nx2_1=2*Nx-1;
Ny2_1=2*Ny-1;
N2x=pow2(ceil(log2(Nx2_1)));
N2y=pow2(ceil(log2(Ny2_1)));

% ---- Indices -----

inx=[0:Nx-1]+Nx;
iny=[0:Ny-1]+Ny;
inxall=1:Nx2_1;
inyall=1:Ny2_1;
indxt=vr.indxt;
indyt=vr.indyt;

% ---- reshaping the vector argument and        ----
% ---- allocating arrays for FFT-multiplication ----
p = reshape(p(:),Nx,Ny,Nz,3);
%vfft=zeros(N2x,N2y);
b=zeros(Ncomp*Nxy*Nzr,1);

%save gr1
%disp('save gr1')
%pause

% ---- multiplication starts ----
for izc=1:Nzr   % ---- source cells ----

   %b3  =zeros(Nm,1);
   %v   =vr.g(izc,:,:,:);
   % pre compute vector p as cell array for Ex, Ey, Ez ;

   for ii=1:Ncomp  % ---- alpha component ----

       b1 = zeros(N2x,N2y);

       % index for which set now we are computing
       ind= (1:Nxy) + (izc-1)*Nxy + (ii-1)*Nxy*Nzr ;

       %       indcell{(izc-1)*Nz+ii} = ind ;

       for izr=1:Nz   % ---- receiver cells ----

           b0 = zeros(N2x,N2y);

           if izc == 1 & ii== 1 ;
               pfft=zeros(N2x,N2y);
               for kk=1:3
                   pfft(1:Nx,1:Ny)=squeeze(p(:,:,izr,kk));
                   % very temp
                   if any(pfft(:))
                       ptmp{kk,izr}=fftgab(pfft,-1);
                   else
                       ptmp{kk,izr}=pfft; ;
                   end
               end
           end

           for jj=1:3  % ---- beta component ----

               %vtmp=v{1,ii,jj,izr}; % OK

               pfft=ptmp{jj,izr}; % OK

               gfft = vr.gfft{izc,ii,jj,izr} ;

               if ( any(pfft(:)) & any(gfft(:)) )
                   %vfft(inxall,inyall)=vr.avg{izc,ii,jj,izr} ;
                   %vfft(indxt,indyt)=vtmp ;

                   %btmp=fftgab(vfft,-1).*pfft ;

                   % btmp=fftgab(fftgab(vfft,-1).*pfft,1);

                   btmp = gfft.*pfft ;

                   b0 = b0 + btmp; % sum up for Z-dir
               end

           end % end of loop for Nz

           b1 = b1 + b0 ; % sum up for beta (jj)

       end % end of loop for beta (jj)

       % now that we are done for dual-sum-up, then apply IFFT
       btmp2 = fftgab(b1,1) ;

       % squezze only we need (actual cell index)

       b2 = btmp2(inx,iny) ;

       % this is at izc-th layer and alpha=ii component
       b(ind) = b2(:) ;
   end

end


if 0
p = reshape(p(:),Nx,Ny,Nz,3);
vfft=zeros(N2x,N2y);
b=zeros(Ncomp*Nxy*Nzr,1);

% ---- multiplication starts ----

for izc=1:Nz   % ---- source cells ----
   ptmp=cell(3,1);
   pfft=zeros(N2x,N2y);
   b2=zeros(Ncomp*Nxy*Nzr,1);
   v = vr.g(:,:,:,izc);

   for kk=1:3
     pfft(1:Nx,1:Ny)=squeeze(p(:,:,izc,kk));
     if any(pfft(:))
       ptmp{kk}=fftgab(pfft,-1);
     end
   end

   for izr=1:Nzr   % ---- receiver cells ----
      for ii=1:Ncomp  % ---- receiver component ----
         b1=zeros(Nx,Ny);
         ind=(1:Nxy)+(izr-1)*Nxy+(ii-1)*Nxy*Nzr;
         for jj=1:3  % ---- source component ----
           pfft=ptmp{jj};
           vtmp=v{izr,ii,jj};
           if ( any(pfft(:)) & any(vtmp(:)) )
               vfft(inxall,inyall)=vr.avg{izr,ii,jj,izc};
               vfft(indxt,indyt)=vtmp;
               btmp=fftgab(fftgab(vfft,-1).*pfft,1);
               b1=b1+btmp(inx,iny);
           end
         end
         b2(ind)=b1(:);
      end
   end
   b=b+b2;
end
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function printout(recpar,da,dn,subtx,ntxset);
% [2004-11-29] add subtx
tmp=[recpar real(dn) imag(dn) real(da) imag(da)];

%num.txset = length(srcpar) / subtx ;
%num.rec   = size(dn,1) / length(srcpar) ;
%tmp = zeros(num.rec * length(srcpar), num.txset);
%subtx
%for ii = num.txset ;
%  for jj = 1 : subtx
 %   tmp(:,ii) = tmp(:,ii) + dn( num.rec * (jj-1) + 1 : num.rec * jj ) ;
 % end;
%end;

%clear dn ;

%dn = tmp(:);

fid=fopen('intout.dat','w');
fprintf(fid,'%+12.7e %+12.7e %+12.7e %2d %2d %9.4e  %+12.7e %+12.7e %+12.7e %+12.7e\n',tmp');
fclose(fid);

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function g2=q2full0(flag,g1)
%   q2full0  take response for a quarter
%           and flip it over (with coefficient)
%           to compute responce on full
%           virtual domain
%           Corrected by Gabor, 29 Nov 1999
%           hxy and hyx components added,
%           and only for 1 z level, 10 Dec 1999
%
%   g2=q2full0(flag,g1)
%
%   flag:   1 - electric 2 - magnetic
%   g1  :   response from a true domain (quarter) g1(Nx,Ny,Nz,3,3)
%   g2  :   response from extended domain, g1(Nx2,Ny2,Nz,3,3)
%
%   amended 11 Jan 2000, ONP
%   magnetic tensor completed 21 Jan 2000, Gabor
%

[Nx,Ny,Nz,Nc,Nt]=size(g1);

if(flag==1) % electric     %           ^ y
c1=[ 1     1    -1 ;       %           |
     1     1    -1 ;       %           |
    -1    -1     1 ;];     %    c2     |
                           %           |
c2=[ 1    -1    -1 ;       %           |
    -1     1     1 ;       % ----------+----------> x
    -1     1     1 ;];     %           |
                           %           |
c3=[ 1    -1     1 ;       %    c1     |    c3
    -1     1    -1 ;       %           |
     1    -1     1 ;];     %           |

else % magnetic            %           ^ y
c1=[ 1     1    -1 ;       %           |
     1     1    -1 ;       %           |
    -1    -1     1 ;];     %    c2     |
                           %           |
c2=[-1     1     1 ;       %           |
     1    -1    -1 ;       % ----------+----------> x
     1    -1     1 ;];     %           |
                           %           |
c3=[-1     1    -1 ;       %    c1     |    c3
     1    -1     1 ;       %           |
    -1     1     1 ;];     %           |

end

if ((Nx==1)|(Ny==1)|(Nz==1))
   Nx2=Nx*2-1;
   Ny2=Ny*2-1;
   ix1=1:Nx-1;  iy1=1:Ny-1;
   ix2=1:Nx-1;  iy2=Ny:Ny2;
   ix3=Nx:Nx2; iy3=1:Ny-1;
   ix4=Nx:Nx2; iy4=Ny:Ny2;

   g2=mirror(mirror(g1,1),2);
   for icr=1:3
   for ict=1:3
      g2(ix1,iy1,:,icr,ict)=c1(icr,ict)*g2(ix1,iy1,:,icr,ict);
      g2(ix2,iy2,:,icr,ict)=c2(icr,ict)*g2(ix2,iy2,:,icr,ict);
      g2(ix3,iy3,:,icr,ict)=c3(icr,ict)*g2(ix3,iy3,:,icr,ict);
   end
   end
   return;
else

   Nx2=Nx*2-1;
   Ny2=Ny*2-1;
   ix1=1:Nx;  iy1=1:Ny;
   ix2=1:Nx;  iy2=Ny:Ny2;
   ix3=Nx:Nx2; iy3=1:Ny;
   ix4=Nx:Nx2; iy4=Ny:Ny2;

   g2=zeros(Nx2,Ny2,Nz,3,3);

   for icr=1:3
   for ict=1:3
       q4=squeeze( g1(:,:,:,icr,ict) );
       g2(ix1,iy1,:,icr,ict)=c1(icr,ict)*flipdim(flipdim(q4,1),2);
       g2(ix2,iy2,:,icr,ict)=c2(icr,ict)*flipdim(q4,1);
       g2(ix3,iy3,:,icr,ict)=c3(icr,ict)*flipdim(q4,2);
       g2(ix4,iy4,:,icr,ict)=q4;
   end
   end
end

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function x = qmrcgstabgab(wordy,vb,ds3,b,m1,m2,x0,phi0);
%
% x = qmrcgstabgab(wordy,vb,ds3,b,m1,m2,x0,phi0);
%
% QMRCGSTAB for forward problem
%
% wordy - if wordy>1:display errors; if wordy<1: no messages
% A     - coefficient matrix
% b     - right hand side
% m1    - vector of diagonal left preconditioner matrix
% m2    - vector of diagonal right preconditioner matrix
% x0    - initial guess
% phi0  - stopping error (relative residual)

Nm = length(ds3);
%Nm = size(A,1);
if length(m1)==0; m1=ones(Nm,1); end;
if length(m2)==0; m2=ones(Nm,1); end;

% ----- integral equation stuff

m1m2 = m1.*m2;
sam2 = ds3.*m2;
b  = m1.*b;
x  = x0./m2;
r  = x.*m1m2 - m1.*multgb1(vb,sam2.*x) - b; % r=M1*A*M2*x-M1*b

% -------------------------------------

mf0=b'*b;
tau=sqrt(r'*r);
rst=r;beta=0;p=0;Ap=0;psi=0;theta=0;eta=0;d=0;
thetatld = 0;etatld=0;
tic;
for it=1:Nm*2

  mfit = it*tau^2;
%  mfit = abs(r'*r);
  err  = sqrt(mfit/mf0);
  if(wordy>1); displayfwdmfit(it,err,toc); end;
  if(err<phi0) break; end;

  p  = r + beta*(p - psi*Ap);

%  Ap = A*p;
  Ap    = p.*m1m2 - m1.*multgb1(vb,sam2.*p);

  alpha = (rst'*r)/(rst'*Ap);

  t = r - alpha*Ap;
%  At = A*t;
  At    = t.*m1m2 - m1.*multgb1(vb,sam2.*t);

  psi = (At'*t)/(At'*At);   % bicgstab
%  psi = (t'*t)/(t'*At);      % bicgstab2

  rold = r;
  r = t - psi*At;

% --- qmr stuff ----
  thetatld = sqrt(t'*t)/tau;
  c = 1/sqrt(1+thetatld*thetatld);
  tautld = tau*thetatld*c;
  etatld = c*c*alpha;
  dtld = p + (theta^2*eta)/alpha*d;
  xtld = x - etatld*dtld;
%  xtld = x + etatld*dtld;

  theta = sqrt(r'*r)/tautld;
  c = 1/sqrt(1+theta*theta);
  tau = tautld*theta*c;
  eta = c*c*psi;
  d = t + (thetatld^2*etatld)/psi*dtld;
  x = xtld - eta*d;
%  x = xtld + eta*d;

% -------------------
  beta = (alpha/psi)*(rst'*r)/(rst'*rold);

end

x=x.*m2;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function buf=readtxt(in)
% READTXT read file into char buffer
% if no file, return empty
%
%  buf= readtxt(in)
%
buf=[];

ff=fopen(in,'r');
if(ff==-1) return; end;

while 1
  line = fgetl(ff);
  if ~isstr(line), break, end
  buf=[buf line sprintf('\n')];
end
fclose(ff);

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function saveasc(where,what,form)
% saveasc save in ascii format
%
%

if(exist('form')~=1) form='%g '; end;

dims=size(what);

switch(length(dims))
  case 0
    return;
  case 2
    d1=dims(1);
    d2=dims(2);
  otherwise
    d1=prod(dims);
    d2=1;
end

what=reshape(what,d1,d2);

in=fopen(where,'w+');

for k1=1:d1
   fprintf(in,form,what(k1,:));
   fprintf(in,'\n');
end

fclose(in);

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function spcest(x,y,z,x1,y1,xyd,solflag);

% Rules for calculating the memory and disk usage of full
% and sparse arrays:
%
%   Array type                Memory(Bytes)
% --------------------------------------------------
% Double array (real)        8*prod(size(A))
% Double array (complex)    16*prod(size(A))
% Sparse array (real)       12*nnz(A)+4*(size(A,2)+1)
% Sparse array (complex)    20*nnz(A)+4*(size(A,2)+1)
%
%   Array type           Binary file size (Bytes)
%                        only one variable/file
% --------------------------------------------------
% Double array (real)        8*prod(size(A))+184
% Double array (complex)    16*prod(size(A))+192
% Sparse array (real)       12*nnz(A)+4*(size(A,2)+1)+204
% Sparse array (complex)    20*nnz(A)+4*(size(A,2)+1)+212
%

Nx = length(x);       % number of cells in x direction inside the body
Ny = length(y);       % number of cells in y direction inside the body
Nz = length(z);       % number of cells in z direction inside the body
Nx1 = length(x1);     % number of elements in x direction to the receivers
Ny1 = length(y1);     % number of elements in y direction to the receivers
Nf = size(xyd,1);     % number of frequencies
Ns = size(xyd,2);     % number of sources
Nzr = size(xyd,3);    % number of different receiver z levels
Ncomp = size(xyd,4);  % number of different field components to be calculated

grmbyte = (2*Nx1-1)*(2*Ny1-1)*Nz*Nzr*Ncomp*3*16/1e6;
etmbyte = Nx*Ny*Nz*3*Nf*Ns/1e6;
if solflag == 1
  gbmbyte = 0;
elseif solflag == 2
  gbmbyte = 9*Nx*Ny*Nz/1e6;
else
  gbmbyte = (2*Nx-1)*(2*Ny-1)*Nz*Nz*9*16/1e6;
end

disp([' ']);
disp('Estimated memory requirements for storing the largest variables:');
disp([' ']);
disp(['Primary electric fields inside the anomalous  body: ' num2str(etmbyte) ' Mbytes']);
disp(['Green tensors from the anomalous body to receivers: ' num2str(grmbyte) ' Mbytes']);
disp(['Scattering Green tensors inside the anomalous body: ' num2str(gbmbyte) ' Mbytes']);
disp([' ']);

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------

function x = succappmodfwd(wordy,vb,en,ds3,sb3,etini,phi0);
% Successive approximation for forward pr with modified green operator

Nm = length(ds3);

sqrtsb3 = sqrt(sb3);
beta    = ds3./(ds3+2*sb3);
maxbeta = max(abs(beta));
coeff = maxbeta/(1-maxbeta);
a = (ds3+2*sb3)./(2*sqrtsb3);

tmp2 = 2*sqrtsb3;
tmp3 = sqrtsb3.*beta;
b  = sqrtsb3.*en;                             % solve for total field

x  = a.*etini;
xold=x;

% ------------ we can erase this stuff after examination -------------
mfit0=abs((b./sqrtsb3)'*(b./sqrtsb3));
mfit0=abs(b'*b);
% --------------------------------------------------------------------

tic;
for it=1:Nm*2

% ------------ we can erase this stuff after examination -------------
  et= x./a;                                  % solve for total field
  rrr  = sqrtsb3.*et - sqrtsb3.*multgb1(vb,ds3.*et) - b;
  mfit = abs(rrr'*rrr);
  err  = sqrt(mfit/mfit0);
  if(wordy>1); displayfwdmfit(it,err,toc); end;
  if(err<phi0) break; end;
% --------------------------------------------------------------------

  x = beta.*x + tmp2.*multgb1(vb,tmp3.*x) + b ; % solve for total field

end

x = x./a;

return;

% ----------------------------------------------------------------------------
% ----------------------------------------------------------------------------


function checkfnxy(FNxy,x,y)
  if length(FNxy)==0;
    if length(x) >= 50 | length(y) >= 50;
      if length(x) >=50 & length(y) >= 50;eFNxy='X and Y';
      elseif length(x) >=50 ; eFNxy='X';
      elseif length(y) >=50 ; eFNxy='Y';
      end
      fprintf('Your discretization in %s direction seems to be lager than default limit 50\n',eFNxy);
      swFNxy = input('Continue ? [1] = Yes, 0 = No  >> ');
      if isempty(swFNxy);swFNxy=1;end;
      switch swFNxy
        case 0
          error('Retry after modification of variable FNxy');
        case 1
          swmodFNxy = input('Modify "FNxy" ? [1] = Yes, 0 = No  >> ');
          if isempty(swmodFNxy);swmodFNxy=1;end;
          switch swmodFNxy
            case 0
              fprintf('Number of cell in %s X/Y will be decreased to 50\n',eFNxy);
            case 1
              FNxy = input('Please input new FNxy >> ');
              if isempty(FNxy);FNxy=[];disp('FNxy is equal to max(Nx,Ny)');end;
          end
      end
    else
      FNxy=[];
    end;
  else
    maxnxy = max([length(x) length(y)]);
    if FNxy < maxnxy ;
      FNxy = [] ;
      disp('FNxy is automatically set to max(Nx,Ny)');
    end
  end;
  return

function checksubfunc(vargin);
  % check for required subfunctions
  p=which('mosvcp');
  if isempty(p);
    error('Cross plat form file copy function MOSVCP is not found');
  end

  p=which('chkmlver');
  if isempty(p);
    error('MATALB version check function CHKMLVER is not found');
  end

  return

function et = fillets(wordy,f,hh,sig_i,an,x,y,z,dz,N,ds_i,sigbg_i,mfit,solflag,srcpar,tmpsave,tth,quickEn,swabse,lqlint,Nsubset,fastEn,subtx)
%
% Function "fillet" has been changed to "fillets" 2004-0820
%
% et = fillets(wordy,f,hh,sig,an,x,y,z,dz,N,ds,sigbg,mfit,solflag,srcpar,tmpsave,tth,quickEn,Nsubset,fastEn);
%
% computing total electric fields inside the body using full IE
%
% wordy   : flag for displaying the status of computation
% f       : vector of frequencies
% hh      : vector of layer thicknesses [] in case of hom.halfspace
% sig_i   : vector of layer conductivities (can be complex)
% an      : vector of layer anisotropies
% x       : vector of x coordinates of the cell centers
% y       : vector of y coordinates of the cell centers
% z       : vector of z coordinates of the cell centers
% dz      : vector of cell sizes in vertical direction
% N       : number of cells to be taken into account horizntally
%          for multiplication by Gb (in this case it is large)
% ds_i    : anomalous conductivity vector
% sigbg_i : anomalous conductivity vector
% mfit    : norm of relative residual for stopping criteria
% solflag : = 1 -> Born approximation
%          = 2 -> QA approximation
%          = 3 -> successive iterations (QA series) for full IE
%          = 4 -> BICGSTAB for full IE
%          = 5 -> QMRCGSTAB for full IE
%          = 6 -> CGMRES(3) for full IE
%          = 7 -> QL approximation
%          = 8 -> QL approximation (different lamda)
%          = 9 -> LQL approximation
%          =10 -> LQL approximation (different lamda)
% srcpar  : cell array of source parameters
% tmpsave : switch of stage 4, added 2004-0729
% tth     : threshold of electric reflectivity tensor Lamda (QL and LQL)
% quickEn : one-time-only background E fields computation (not general, not recomended but quick)
% fastEn  : compute only sparse En and interpolate to fine En (not recomended but fast)
% et      : struct array of total electric fields
% subtx   : number of sub-txs in each tx

Nf = length(f);
Nx = length(x); Ny = length(y); Nz = length(z);
Nxy= Nx*Ny; Nxyz=Nxy*Nz ;

if (iscell(sig_i))
  kcomp1 = 2;
else
  kcomp1 = 0;
  sig = sig_i;
  sigbg = sigbg_i;
end

if (iscell(ds_i))
  kcomp2 = 2;
else
  kcomp2 = 0;
  ds = ds_i;
end

if(Nx==1);dx=dz(1);else;dx=x(2)-x(1);end
if(Ny==1);dy=dz(1);else;dy=y(2)-y(1);end

Nsrc=length(srcpar);
[xm,ym,zm]=ndgrid(x,y,z);
xm=xm(:); ym=ym(:); zm=zm(:);

ds3=repmat(ds,3,1);
sb3=repmat(sigbg,3,1);

if solflag == -1 ;
  indzero = find( ds3 == unique( sb3 ) );
end

[m1,m2]=mkprecnd(ds3,sb3); % preconditioners

et.e=cell(Nf,Nsrc);
enold=cell(Nsrc,1);

% load finer grid data for QL and LQL
if solflag >= 7 ;
  load fwdstg1f.mat;
  [Nxf,Nyf,Nzf,Nxyf,Nxyzf,xmf,ymf,zmf,dxf] = load_fine_grid_pars(xf,yf,zf);
end

% load electric fields computed in "solflag= 7 or 9"
if solflag == 8;
  load eqltmp;
elseif solflag == 10 ;
  load elqltmp;
end

if solflag <= 8 ; Nsubset = 0 ; end;

if Nsubset == 0;
  Ngrp = 1;
elseif (solflag == 9 | solflag == 10) & Nsubset~=0;
  Ngrp = fix( Nsrc / Nsubset ) ;
  if mod( Nsrc , Nsubset ) ~= 0;
    Ngrp = Ngrp + 1;
  end
else
  Ngrp = 1;
end

Nsrc_org = Nsrc;

if solflag <=6 & subtx ~= 1;
  Nsrc = Nsrc / subtx ;
end

% ----------------------------------------------
N3d = struct('Nx',Nx,'Ny',Ny,'Nz',Nz,'Nxyz',Nxyz);
if solflag >= 7;
  N3d.Nxf=Nxf; N3d.Nyf=Nyf; N3d.Nzf=Nzf; N3d.Nxyzf=Nxyzf;
end;
if wordy >=1; show_calculation_config(solflag,N3d) ; end ;

% ----------------------------------------------
% Main part of the FILLETS.M
% There are 3 sub stages
% ----------------------------------------------

for ii = 1 : Nf ; % loop for frequency
  %
  if (kcomp1 == 2);sig = sig_i{ii}; sigbg = sigbg_i{ii}; end
  if (kcomp2 == 2);ds = ds_i{ii}; end

  % ----------------------------------------------
  % Sub stage 1 :
  % ----------------------------------------------

  for kk = 1 : Ngrp ; % loop for Tx subset (LQL only)

    % for LQL with Tx subset
    if (solflag == 9 | solflag == 10) & Nsubset ~= 0 ;
      if (mod(Nsrc_org,Nsubset)~= 0) & (kk == Ngrp);
        Nsrc = mod(Nsrc_org,Nsubset); % last Tx subset
      else ;
        Nsrc = Nsubset ; % regular Tx subset
      end
      fprintf('Now computing group #%g\n',kk);
    end
    if solflag==9;
      show_lql_config(Ngrp,kk,Nsrc, Nsubset);
    end
    ena = zeros(3*Nxyz,1);
    enaorg = ena;
    %
    % sub stage 1
    % compute and prepare backgrounf En and Grenn tensor coefficient
    %
    if quickEn == 1 & solflag >=7 ; % quick En and QL/LQL
      if solflag <= 6 ;
        error('This option is only for solflag>=7');
      end

      % pre-process if quickEn
      quickdir = 1 ;% x

      dtx = abs(srcpar{2}(3)-srcpar{1}(3)) ;

      % For QL/LQL calculation
      if solflag == 7 | solflag ==9 ;

        for iq = Nsubset*(kk-1)+1 : Nsubset*(kk-1)+Nsrc;
          x_quik(iq,:) = str2num(sprintf('%12.7e ',(x-(iq-1)*dtx)));
        end
        x1 = (unique(x_quik(:))) ;
        x1 = str2num(sprintf('%12.7e ',x1)) ;
        if fastEn == 0;
          for iq = Nsubset*(kk-1)+1:Nsubset*(kk-1)+Nsrc
            xf_quik(iq,:) =str2num(sprintf('%12.7e ',(xf-(iq-1)*dtx)));
          end
          x1f = (unique(xf_quik(:)))
          x1f=str2num(sprintf('%12.7e ',x1f))
        end
        y1 = y ; z1 = z;
        [xm,ym,zm] = ndgrid(x1,y1,z1);
        xm = xm(:); ym = ym(:); zm = zm(:);
        vb = fillgb1(wordy,f(ii),hh,sig,an,x,y1,z1,dz,N);
        % get source parameters
        ssrrcc=srcpar{1};
        % compute En with coarse grid
        if ssrrcc(1) == 1; % plane wave
          en1 = green3d(f(ii),hh,sig,an,xm,ym,zm,1);
          en1 = reshape(en1,Nxyz,3);
          if ssrrcc(2)==1; en1(:,[2 3])=0;end;
          if ssrrcc(2)==2; en1(:,[1 3])=0;end;
        else
          en1 = green3d(f(ii),hh,sig,an,xm,ym,zm,ssrrcc);
        end
        encxyz = reshape(en1,length(x1),Ny,Nz,3);

        % compute En with fine grid
        % this is obsolete
        % preparation for fine grid
        %x1f=[-xf(end) : dxf : xf(end)]; % new X coordinates old
        y1f = yf ; z1f = zf;
        [xmf, ymf, zmf] = ndgrid(x1f ,y1f, z1f);
        xmf = xmf(:); ymf = ymf(:); zmf = zmf(:);
        switch fastEn
          case 0 ; % compute En on fine using GREEN3D
            if ssrrcc(1)==1
              enf1 = green3d(f(ii),hh,sig,an,xmf,ymf,zmf,1);
              enf1 = reshape(enf1,Nxyzf,3);
              if ssrrcc(2)==1; enf1(:,[2 3])=0;end;
              if ssrrcc(2)==2; enf1(:,[1 3])=0;end;
            else
              enf1 = green3d(f(ii),hh,sig,an,xmf,ymf,zmf,ssrrcc);
            end
            enfxyz = reshape(enf1,length(x1f),Nyf,Nzf,3);
          case 1 ; % interpolate En from En on sparse
           ;
        end
      end % end of solflag == 7 or 9


      % else, quickEn is off or Born and Full IE
    elseif quickEn == 0 | solflag <= 6;

      if solflag == -1 | solflag == 0 | solflag == 1 | solflag == 8 | solflag == 10;
        % nothing to do
      elseif solflag == 2 ; % QA Approximation
        g = zeros(Nxy,Nz,3,3);
        for jj = 1 : Nz
          gtmp = green3d(f(ii),hh,sig,an,0,0,z(jj),[-2 dx dy dz(jj) 0 0 z(jj)]);
          gtmp = reshape(gtmp,1,3,3);
          g(:,jj,:,:) = repmat(gtmp,Nxy,1);
        end
        g = reshape(g,Nxyz,3,3);
      else % prepare Green's coefficient for sparse grid
        vb = fillgb1(wordy,f(ii),hh,sig,an,x,y,z,dz,N);
      end
    end % end of substage 1

    % --------------------------------------------------------------------------
    %
    % sub stage 2
    %
    % repeat for the number of Tx
    %
    for jj = Nsubset*(kk-1)+1 : Nsubset*(kk-1)+Nsrc
      %  for jj = 1 : Nsrc
      %pause;
      if (solflag ~= 8) & (solflag ~= 10) ;
        %
        % quickEn is On, just pick up current En from quickEn
        %
        if quickEn == 1 & solflag >= 7;
          % pick up En at coarse grid
          %en = encxyz(end-jj-2*Nx+2+1:2:end-(jj-1), : , :, :);%old
          indxqk = findin(x1,x_quik(jj,:));
          %indxqk = find(x1==x_quik(jj,:));
          indxqk = (indxqk(find(indxqk~=inf))) ;
          en = encxyz(indxqk, : , :, :);
          en = en(:);
          enorg = en;
          enquick{ii,jj} = en(:);
          % pick up En at fine grid
          if fastEn == 1;
            [enf]= lamdafine(en,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,1);
          elseif fastEn == 0;
            indxfqk = findin(x1f,xf_quik(jj,:));
            indxfqk = (indxfqk(find(indxfqk~=inf))) ;
            enf = enfxyz(indxfqk, : , :, :);
          end;
          enff{ii,jj} = enf(:);
          %
        elseif quickEn == 0 | solflag <=6;
          % else, quickEn is off, compute En for each corresponding Tx
          ssrrcc=srcpar{jj};
          % plane wave
          if ssrrcc(1)==1;
            en = green3d(f(ii),hh,sig,an,xm,ym,zm,1);
            en = reshape(en,Nxyz,3);
            if ssrrcc(2)==1; en(:,[2 3])=0;end;
            if ssrrcc(2)==2; en(:,[1 3])=0;end;
            % artificial source �ξ��
          else
            if subtx == 1 ; % regular single transmitter (w/o combined transmitter)
              en = green3d(f(ii),hh,sig,an,xm,ym,zm,ssrrcc);
            else % combined multiple transmitter
              en = 0;
              for i_st = ( (jj-1)*subtx + 1) : (jj * subtx) ;
                % main tx index=jj >> current sub tx index = i_st
                % sum up all tx in this subset and generate psued normal E fields
                en = en + green3d(f(ii),hh,sig,an,xm,ym,zm,srcpar{i_st});
              end
            end;
          end
          en = en(:);
          ennoquick{ii,jj} = en(:);
        end % end of quickEn

          % if LQL, we need to sum up all En

        if solflag == 9; % sum of En for LQL
          enorg = en ; % keep en without ABS
          enorgc{ii,jj} = en ; % this will be used as input data for En on fine grid
          if swabse == 1 ;
            en = abs3all(en,Nxyz); % here, we take ABS value for LQL
          end
          ena = ena + en; % ABS
          enaorg = enaorg + enorg; % without ABS
        end

      end % end of solflag ~= 8 nor 10
      %
      %-------------------------------------------------------
      %
      % sub stage 3
      % compute "et" inside anomalous domain
      %
      if solflag == 0 ;% use FD result
        load ea;
        et1 = en + conj(ea) ;
      elseif solflag == -1 ;%
        % take off
      elseif  solflag == 1;% Born approximation
        et1 = en;
      elseif solflag == 2;% QA approximation
        eb3 = reshape(en,Nxyz,3);
        tmp11=0;
        for kk = 1:3
          for ll = 1:3
            tmp11 = tmp11 + conj(eb3(:,kk)).*g(:,kk,ll).*eb3(:,ll);
          end
        end
        C=tmp11./sum((abs(eb3).^2),2);
        tmp12= 1./(1-C.*ds);
        et1 = en.*repmat(tmp12,3,1);
      else;
        %
        % solflag = 3    : QA series
        %         = 4-6  : full IE
        %         = 7,8  : QL
        %
        % if solflag <= 7, SLEs should be solved
        %
        % here, we start with the preparation of initial guess
        % for the SLEs based on previous frequency's result
        %
        if solflag <=7 ;
          if ii>1     % computing ini. guess from previous freq.
            if solflag <= 6
              tmp1=et.e{ii-1,jj};
            elseif solflag == 7
              tmp1=eqltmp.et{ii-1,jj};
            end
            tmp2=reshape(en,Nxyz,3);
            tmp3=reshape(enold{jj},Nxyz,3);
            %whos tmp1 tmp2 tmp3
            etini=tmp1.*repmat(sum(abs(tmp2),2)./sum(eps+abs(tmp3),2),3,1);
          else
            etini = zeros(length(sb3),1);
          end
          enold{jj} = en;
        end
        %
        % Let's start to solve the SLEs
        %
        % just save En for future usage [2004-11-23]
        %      save en en ;

        if solflag == 3
          et1 = succappmodfwd(wordy,vb,en,ds3,sb3,etini,mfit);
        elseif solflag == 4
          et1 = bicgstgab(wordy,vb,ds3,en,m1,m2,etini,mfit);
        elseif solflag == 5
          et1 = qmrcgstabgab(wordy,vb,ds3,en,m1,m2,etini,mfit);
        elseif solflag == 6
          % et1 = gmrgab(wordy,vb,ds3,en,m1,m2,etini,mfit,3);
          et1 = gmrgab(wordy,vb,ds3,en,m1,m2,etini,mfit,20);
          %eaqlf = et1 - en;
        elseif solflag == 7 | solflag == 8; % QL 2004-0819
          %
          % for this "solflag", background fields on finer grid should be also computed
          %
          if solflag == 7;
            if quickEn == 0
              ssrrcc=srcpar{jj}; % pick up src#
              if ssrrcc(1)==1
                enf = green3d(f(ii),hh,sig,an,xmf,ymf,zmf,1);
                enf = reshape(enf,Nxyzf,3);
                if ssrrcc(2)==1; enf(:,[2 3])=0;end;
                if ssrrcc(2)==2; enf(:,[1 3])=0; end;
              else;
                enf = green3d(f(ii),hh,sig,an,xmf,ymf,zmf,ssrrcc);
              end;
            end
            enf = enf(:);
            % solve the SLEs for total EQL using GMRM

            etql = gmrgab(wordy,vb,ds3,en,m1,m2,etini,mfit,20) ;

            % get the anomalous E_QL by subtracking normal field
            eaql = etql - en ;
            % new option to save en and eaql
            % these variavle will be used for solflag=8 | 10
            eqltmp.et{ii,jj} = etql;
            eqltmp.enc{ii,jj} = en;
            eqltmp.enf{ii,jj} = enf;
            eqltmp.eaqlc{ii,jj} = eaql;
          elseif solflag == 8 % just solve using different threshold
            % set current background fields (on fine and coarse grid)
            % and QL anomalous fields on the coarse grid
            en = eqltmp.enc{ii,jj};
            enf = eqltmp.enf{ii,jj};
            eaql = eqltmp.eaqlc{ii,jj};
          end; % end of solflag ==7

          % -----------------------------------------------------------
          % compute refrectivity tensor

          if swabse==1
            % here, we take ABS value
            [en] = abs3all(en,Nxyz);
            [enfabs] = abs3all(enf,Nxyzf);
          else
            enfabs = enf;
          end

          % compute lamda with sparse grid
          lamda = en.^(-1) .* eaql ;
          %ndsing = unique(find(abs(real(en))<=tth | abs(imag(en))<=tth));
          indsing = unique(find(abs(en) <= tth));
          lamda(indsing) = 0;
          if(wordy);
              fprintf('# of abs(En) <= %g is %g \n',tth,length(indsing));
              fprintf('Lamda of these cells are assumed to be 0 \n');
          end
          %
          % important change 2004-0906
          % no interpolation of lamda but just re-distribution
          %
          [lamdaf] = lamdafine(lamda,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,0);
          %
          % -----------------------------------------------------------
          % compute Ea_QL for finer grid
          eaqlf = lamdaf .* enfabs ;
          % eaqltmp = lamda .* en ;
          % compute Et_QL for finer grid
          %
          et1 = eaqlf + enf;
          %
          % end of QL approximation

        elseif solflag == 9;
          % here, nothing to do for LQL
        elseif solflag == 10;
          % here, nothing to do for LQL
        else;
          error('Solution flag must be 1,2,3,4,5,6,7(QL) or 8(QL).');
        end; % end of solvers for solflag >=3
      end; % end of solvers inside each SRC loop

      if solflag <=8
        %      save et1fl et1;

        et.e{ii,jj} = et1;
        if solflag <=6 & solflag ~=0;
          et.ea{ii,jj} = et1- en;
        elseif solflag >6;% for QL
          %        et.ea{ii,jj} = et1- enf;
          et.ea{ii,jj} = eaqlf;
        end;
        if (wordy>0); disp(['Calculation of total electric field is done for f = ' num2str(f(ii)) ' Hz and src # ' num2str(jj)]);end;
        % save intermediate result 2004-0728
        if tmpsave == 1;
          etmp{ii,jj} = et1;
          et1InterName=['et1_int_',num2str(ii),num2str(jj),'.mat'];
          save et1tmp etmp ii jj;
        end
      end
    end % end of Nsrc loop

    clear enfxyz encxyz; % 2004-1005
    if wordy>=2; disp('saving et and en');end

    % ----------------------------------------------------------------
    %
    % in the case of LQL
    %
    % ----------------------------------------------------------------

    if solflag == 9 | solflag ==10 ;
      if solflag == 9 ;
        en = ena / Nsrc ; % abs(En)_avg
        enorgav = enaorg / Nsrc ; % original En_avg
        %en = abs3all(enorgav,Nxyz) / Nsrc;
        %elqltmp.en = en;
        %elqltmp.enorgav = enorgav;
      elseif solflag == 10;
        enff = elqltmp.enff;
        en = elqltmp.en ;
        enorgav = elqltmp.enorgav;
      end
      %
      %
      % solve the SLEs for total EQL using GMRM
      etini = zeros(length(sb3),1);
      if wordy>=2; fprintf('Now solving the SLES for specified subset...\n');end;
      etql = gmrgab(wordy,vb,ds3,enorgav,m1,m2,etini,mfit,20) ;
       % get the anomalous E_QL by subtracking normal field
      if(wordy);fprintf('Solving the SLES is done');end;
      eaql = etql - enorgav ;
      %
      %if solflag == 9 ;
      %      elqltmp.enc{ii,jj} = en;
      %      elqltmp.eaqlc{ii,jj} = eaql;
      %end
      %
      % compute refrectivity tensor
      %
      % 2004-0906 new implementation for LQL
      % using interpolated En and Ea(LQL) and then compute lamdaf
      [eaqlfint] = lamdafine(eaql,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,0);

      if lqlint == 1 ; % lamdaf just multipulication of En^-1 and EaLQL
        [enfint]= lamdafine(en,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,1);
        lamdaf = enfint.^(-1) .* eaqlfint ;
        indsing = unique(find(abs(enfint)<=tth));
        lamdaf(indsing) = 0;
      elseif lqlint == 0 ; % lamdaf after interpolation of lamda
        lamda = en.^(-1) .* eaql ;
        indsing = unique(find(abs(en)<=tth));
        lamda(indsing) = 0;
        [lamdaf]= lamdafine(lamda,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,0);
      end

      if wordy>= 2;
          fprintf('# of abs(En) <= %g is %g \n',tth,length(indsing));
          fprintf('Lamda of these cells are assumed to be 0 \n');
      end;
      %[lamdaf] = lamdafine(lamda,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf);

      for jj = Nsubset*(kk-1)+1 : Nsubset*(kk-1)+Nsrc

        % in the case of quickEn=0, En for fine grid will be calculated here
        if quickEn == 0 ;
          switch fastEn
            case 0 % compute using GREEN3D
              ssrrcc=srcpar{jj};
              if ssrrcc(1)==1
                enf = green3d(f(ii),hh,sig,an,xmf,ymf,zmf,1);
                enf = reshape(enf,Nxyzf,3);
                if ssrrcc(2)==1; enf(:,[2 3])=0;end;
                if ssrrcc(2)==2; enf(:,[1 3])=0; end;
              else;
                enf = green3d(f(ii),hh,sig,an,xmf,ymf,zmf,ssrrcc);
              end;

            case 1 % interpolate form the En with sparse grid
              enf = lamdafine(enorgc{ii,jj},Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,1);
          end
          enf = enf(:);
          enff{ii,jj} = enf(:);
        end

        % compute Ea_QL for finer grid
        if swabse==1
          % here, we take ABS value
          [enffabs] = abs3all(enff{ii,jj},Nxyzf);
          eaqlf = lamdaf .* enffabs;
        else
          eaqlf = lamdaf .* enff{ii,jj};
          %        eaqlf = lamdaf .* enf;
        end

        % compute Et_QL for finer grid
        et1 = eaqlf + enff{ii,jj};
        % and store Et_QL
        et.e{ii,jj} = et1;
        et.ea{ii,jj} = eaqlf;
        fprintf('ii = %g, jj = %g \n',ii,jj);
        if (wordy>0); disp(['Calculation of total electric field is done for f = ' num2str(f(ii)) ' Hz and src # ' num2str(jj)]);end;
        if tmpsave == 1;
          etmp{ii,jj} = et1;
          et1InterName=['et1_int_',num2str(ii),num2str(jj),'.mat'];
          save et1tmp etmp ii jj;
        end

      end ; % end of LQL Tx set loop (jj)
      clear en ena;
    end ;% end of solflag = 9 or 10

  end ; % end of LQL grp.

end ; % end of Nf loop



% ----------------------------------------------------------------------------

function chkenbhv(en,Nxyz,Nx,Ny,Nz,x,y,z,sc,enabs)
  switch sc;
    case 1;
      eComp = 'n';
    case 2;
      eComp = 'a';
    case 3;
      eComp = 't';
  end


  end3.all = reshape(en,Nxyz,3);
  end3.x = end3.all(:,1);
  end3.y = end3.all(:,2);
  end3.z = end3.all(:,3);

  sen=sort(abs(en));
  sen3=sort(abs(end3.all));
  sen3.x=sort(abs(end3.x));
  sen3.y=sort(abs(end3.y));
  sen3.z=sort(abs(end3.z));
  inde = [1:Nxyz];

  sen3abs = sort(enabs(1:Nxyz));

  end3.x3 = reshape(end3.x,Nx,Ny,Nz);
  end3.y3 = reshape(end3.y,Nx,Ny,Nz);
  end3.z3 = reshape(end3.z,Nx,Ny,Nz);

  end3.x3r = real(end3.x3);
  end3.y3r = real(end3.y3);
  end3.z3r = real(end3.z3);

  end3.x3i = imag(end3.x3);
  end3.y3i = imag(end3.y3);
  end3.z3i = imag(end3.z3);



  [xc2,yc2] = ndgrid(x,y);

  figure(2)
  for i=1:Nz;
    subplot(Nz,3,(i-1)*3+1);
    contourf(xc2,yc2,end3.x3r(:,:,i));
    axis tight;colorbar;
    if i~=Nz;set(gca,'XTickLabel',{''});
    else;
      switch sc;
        case 1;
          xlabel('E^{n}_x');
        case 2;
          xlabel('E^{a}_x');
        case 3;
          xlabel('E^{t}_x');
      end
    end;

    subplot(Nz,3,(i-1)*3+2);
    contourf(xc2,yc2,end3.y3r(:,:,i));
    axis tight;colorbar;
    if i~=Nz;set(gca,'XTickLabel',{''});
    else;
      switch sc;
        case 1;
          xlabel('E^{n}_y');
        case 2;
          xlabel('E^{a}_y');
        case 3;
          xlabel('E^{t}_y');
      end
    end;

    subplot(Nz,3,(i-1)*3+3);
    contourf(xc2,yc2,end3.z3r(:,:,i));
    axis tight;colorbar;
    if i~=Nz;set(gca,'XTickLabel',{''});
    else;
      switch sc;
        case 1;
          xlabel('E^{n}_z');
        case 2;
          xlabel('E^{a}_z');
        case 3;
          xlabel('E^{t}_z');
      end
    end

  end

function [en] = abs3all(en,Nxyz)
  aen.x = en(1:Nxyz);
  aen.y = en(Nxyz+1:2*Nxyz);
  aen.z = en(2*Nxyz+1:end);

%  entmp = sqrt(real(aen.x).^2 + real(aen.y).^2 + real(aen.z).^2 +imag(aen.x).^2 + imag(aen.y).^2 + imag(aen.z).^2);
  entmp = sqrt(abs(aen.x).^2 + abs(aen.y).^2 + abs(aen.z).^2 );
  en = [entmp;entmp;entmp];


function [outData] = replacedge(inData);

  sizeData = size(inData);
  n.Data = length(inData(:));
  n.X = sizeData(1);
  n.Y = sizeData(2);
  n.Y = sizeData(3);

  outData = inData;

  % replace Z direction edge face
  outData(:,:,1) = inData(:,:,2);
  outData(:,:,end) = inData(:,:,end-1);

  % replace X direction edge face
  outData(1,:,:) = inData(2,:,:);
  outData(end,:,:) = inData(end-1,:,:);

  % replace Y direction edge face
  outData(:,1,:) = inData(:,2,:);
  outData(:,end,:) = inData(:,end-1,:);


% ----------------------------------------------------------------------------

function I=findin(A,b)
% A is a matrix and b is a column vector
% This function returns a column vector of indices I
% such that A(I)==b if the elements of b are in A.
% If b(k) is not in A, then I(k) is Inf.
%
% Written by Steve Lord (slord@mathworks.com)
% with input/suggestions from Tim Burke
%
% Download this file from MATLAB Central
%   (http://www.mathworks.com/matlabcentral)

Av=A(:);
Bv=b(:);
Q1=kron(Av',ones(size(Bv)));
Q2=kron(ones(size(Av')),Bv);
I=(Q1==Q2).*kron(1:length(Av),ones(size(Bv)));
I(I==0)=Inf;
I=min(I')';


function [srcpar,sig0,hh0,an0,x,y,z,dz,                ...
          kcomp,sigc0,chrgabl0,timeconst0,freqconst0,  ...
          mfit, solflag, stg, wordy, FNxy, tmpsave, ...
          tth, quickEn, swabse, lqlint, Nsubset, fastEn, subtx, combo] ...
        = read_intem3d_par(nargin,varargin,p);

    p={' ',' '};
    p=buf2par(p,arg2buf(nargin,varargin));
    p=buf2par(p,readtxt('intem3d.par'));
    srccnt=0;
    for kk=1:size(p,1);
      if length(p{kk,1})>=8 & p{kk,1}(1:6)=='srcpar';
        srccnt=srccnt+1;
        inl=find(p{kk,1}=='{');
        inu=find(p{kk,1}=='}');
        srcin(srccnt)=str2num(p{kk,1}((inl+1):(inu-1)));
      end;
    end
    numsrc=max(srcin(:));
    for kk=1:length(srcin)
      srcpar{srcin(kk)}=getpar(p,'double',['srcpar{' num2str(srcin(kk)) '}'],' ',' ');
    end
    % add kcomp (2004-0724)
    kcomp=[]; sigc0=[]; chrgabl0=[]; timeconst0=[]; freqconst0=[];
    kcomp=getpar(p,'double','kcomp', '','');
    an0=getpar(p,'double','an0', '','');
    hh0=getpar(p,'double','hh0', '','');
    mfit=getpar(p,'double','mfit', '','');
    sig0=getpar(p,'double','sig0', '','');
    % add IP parameters 2004-0728
    %
    % kcomp is a switch of IP parameters
    % kcomp(1) for 1D background
    % kcomp(2) for 3D anomalous domain
    %
    % 0 : no IP/complex effect
    % 1 : with real and complex conductivity
    % 2 : with Cole-Cole patameters
    %
    % add kcomp (2004-0724)
    if length(kcomp)==0;
      kcomp=[0 0]; % default is no complex conductivity
      %disp('Variable kcomp is defined as default 0 (Off). ');
    end;

    switch kcomp(1)
      case 2;
        chrgabl0=getpar(p,'double','chrgabl0', '','')
        timeconst0=getpar(p,'double','timeconst0', '','');
        freqconst0=getpar(p,'double','freqconst0', '','');
      case 1;
        sigc0=getpar(p,'double','sigc0', '','');
    end
    % end of IP parameters

    solflag=getpar(p,'double','solflag', '','');
    stg=getpar(p,'double','stg', '','');
    wordy=getpar(p,'double','wordy', '','');

    x=getpar(p,'double','x', '','');
    y=getpar(p,'double','y', '','');
    z=getpar(p,'double','z', '','');
    dz=getpar(p,'double','dz', '','');
    FNxy=getpar(p,'double','FNxy', '','');

    % new switch for saveing intermediate result in the Stg.2 2004-0729
    % 0: Off,  1:On
    % This switch help for multi source or frequency problem
    tmpsave=getpar(p,'double','tmpsave', '','');
    %sn=getpar(p,'double','sn', '','');
    subtx=getpar(p,'double','subtx', '','');
    combo=getpar(p,'double','combo', '','');

    %
    %  new parameters xf, yf, zf and dzf
    % these are finer grids than x, y, z and dz
    %

    % check input parameters
    %
    % modified from 'isempty' to 'length' 2004-0730
    %
    if length(an0)==0; error(' Variable an0 is not defined. '); end;
    if length(hh0)==0 & length(sig0)>2; error(' Variable hh0 is not defined. '); end;
    if length(mfit)==0; error(' Variable mfit is not defined. '); end;
    if length(sig0)==0; error(' Variable sig0 is not defined. '); end;
    if length(solflag)==0; error(' Variable solflag is not defined. '); end;
    if length(srcpar)==0; error(' Variable srcpar is not defined. '); end;
    if length(stg)==0; error(' Variable stg is not defined. '); end;
    if length(wordy)==0; error(' Variable wordy is not defined. '); end;
    if length(x)==0; error(' Variable x is not defined. '); end;
    if length(y)==0; error(' Variable y is not defined. '); end;
    if length(z)==0; error(' Variable z is not defined. '); end;
    if length(dz)==0; error(' Variable dz is not defined. '); end;
    % modified 2004-0818
    % check variable FNxy
    if 0 % check and set FNxy based on the 3D anomalous domain (now, comment out)
      checkfnxy(FNxy,x,y);
    end;
    if length(FNxy)==0;
      FNxy = inf ;
      % bug fix for intem3d3 by Ken Yoshioka
      % old default FNxy is fixed as 50 but this is too small in some cases
      % so that he suggested and decided to use FNxy = inf instead of 50.
      %
    end

    %add tmpsave (2004-0729)
    if length(tmpsave)==0; tmpsave=0; if solflag>70;disp('Variable tmpsave is defined as default 0 (off). ');end; end;
    %if length(sn)==0; sn=1;disp('Variable sn is defined as default 1 (One box). '); end;
    if length(combo)==0; combo=2; end;
    % pre compute complex resistivity and conductivity for 1D background 2004-0728
    recparip='recpar.dat';
    if (exist(recparip)==2)
     tmp=loadasc(recparip);
    else
     error('recpar.dat does not exist.')
    end
    f=tmp(:,end);
    if kcomp(1)==2 | kcomp(2)==2;
      if length(unique(f))~=1;
        disp('Your receiver file contains more than two freqencies.');
        error('For this version, only one frequency is avairable for Cole-Cole model');
      end;
    end;
    switch kcomp(1) ; % preparation of 1D background conductivities
     case 0;
      sig0 = sig0;
     case 1
      % check and compute complex conductivity with real and complex input data
      if length(sigc0)==0; error(' Variable sigc0 is not defined. '); end;
      if length(sig0)~=length(sigc0);error(' Variable sigc0 should have same size for sig0');end
      sig0 = sig0 + 1i*sigc0;
     case 2 ;
      if length(chrgabl0)==0; error(' Variable chrgabl0 is not defined. '); end;
      if length(timeconst0)==0; error(' Variable timecosnt0 is not defined. '); end;
      if length(freqconst0)==0; error(' Variable freqconst0 is not defined. '); end;
      %
      % for the Cole-Cole model, complex conductivity will be computed later part
      % 2004-0730 added 1D complex conductivity
      %
      lcolecole=[length(sig0) length(chrgabl0) length(timeconst0) length(freqconst0)] - length(sig0);
      if any(lcolecole); error('All Cole-Cole parameters shoule be same size');end;
    end

    if length(subtx)==0; subtx=1; end %disp('Variable subtx is set to 1'); end;
    % check for subtx. this parameter is just under testing and vaild
    % for only solflag <= 6
    if subtx == 0; subtx = numsrc ; end ; % subtx = 0, sum up all Txs
    if (subtx ~= 1) & (mod(numsrc,subtx) ~= 0) ;
      error('Number of input SRCPAR should be devided by SUBTX');
    end;
    if (subtx ~= 1) & (solflag > 6) ;
      error('Option SUBTX is valid for only solflag <= 6');
    end

    % This is temporaly setting
    if solflag==8 ; solflag=9 ; end;

    % special parameters for QL/LQL
    tth=getpar(p,'double','tth', '','');
    quickEn=getpar(p,'double','quickEn', '','');
    swabse=getpar(p,'double','swabse', '','');
    lqlint=getpar(p,'double','lqlint', '','');
    Nsubset=getpar(p,'double','subset', '','');
    fastEn=getpar(p,'double','fastEn', '','');

    if length(tth)    ==0; tth=1e-15 ; if solflag>=70;disp('Variable tth is defined as default 1e-7. '); end; end;
    if length(quickEn)==0; quickEn=0 ; if solflag>=70;disp('Variable quickEn is set to 0'); end; end;
    if length(swabse) ==0; swabse =1 ; if solflag>=70;disp('Variable swabse is set to 1') ; end; end;
    if length(lqlint) ==0; lqlint =1 ; if solflag>=70;disp('Variable lqlint is set to 1') ; end; end;
    if length(Nsubset)==0; Nsubset=0 ; if solflag>=70;disp('Variable Nsubset is set to 0'); end; end;
    if length(fastEn) ==0; fastEn =0 ; if solflag>=70;disp('Variable fastEn is set to 0') ; end; end;

    % This is temp. setting
    % For LQL but subset=1 (means QL), to fit QL default, lqlint is set to 0
    if solflag== 9 & Nsubset==1; lqlint = 0 ; end;

% ----------------------------------------------------------------------------
function [sigma_comp] = get_background_layer_sig(kcomp,sig0,sigc0,chrgabl0,timeconst0,freqconst0)
% ----------------------------------------------------------------------------
f = get_frequency_recpar;

% 2004-0730 added 1D complex conductivity
switch kcomp(1)
  case 0;
    sigma_comp = sig0;
  case 1
    sigma_comp = sig0 + 1i*sigc0;
  case 2 ;
    [res_comp,sigma_comp]=ipp2csig(sig0.^(-1),chrgabl0,timeconst0,freqconst0,unique(f)');
end


% ----------------------------------------------------------------------------
function [tmp] = get_recpar
% ----------------------------------------------------------------------------
% Get frequency information
recparip='recpar.dat';
if (exist(recparip)==2)
  tmp=loadasc(recparip);
else
  error('recpar.dat does not exist.')
end



% ----------------------------------------------------------------------------
function [freq] = get_frequency_recpar
% ----------------------------------------------------------------------------
[tmp] = get_recpar;
freq=tmp(:,end);

return



% ----------------------------------------------------------------------------
function b = fwdrec_ip(wordy,f,hh,sig_i,an,x,y,z,dz,N,zr,dind,indbody,m_i,et,wi);
% ----------------------------------------------------------------------------
% b = fwdrec_ip(wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody,m,et,wi);
%
% Calculates the anomalous fields at the receivers
%
% wordy,f,hh,sig,an,x,y,z,dz,N,zr,dind,indbody : see fillgr1.m
% b    : Nd,1 vector of anomalous fields at the receivers
%        Nd is the number of data
%
% m    : Nxyz,1 anomalous conductivity vector, Nxyz is the
%         number of model parameters (cells)
%
% et   : struct array of total electric fields inside the anomalous body
%
% wi   : struct array of interpolation matrices, irregular
%        data indices, data and body parameters.
%         (see fillwi.m for details)
if (iscell(sig_i))
  kcomp1 = 2;
else
  kcomp1 = 0;
  sig = sig_i;
end

if (iscell(m_i))
  kcomp2 = 2;
else
  kcomp2 = 0;
  m = m_i;
end

Nf = wi.Nf;
Nsrc = wi.Nsrc;
Nzr = wi.Nzr;
Ncomp = wi.Ncomp;
Nxy = wi.Nxy;
Nd=max(wi.indu(:));

b=zeros(Nd,1);

for ii = 1:Nf

  if (kcomp1 == 2)
    sig = sig_i{ii};
  end
  if (kcomp2 == 2)
    m = m_i{ii};
  end
  vr = fillgr1(wordy,f(ii),hh,sig,an,x,y,z,dz,N,zr,dind,indbody);
  for jj = 1:Nsrc
    t=multa1(m,et.e{ii,jj},vr) ;
    %save tmpall2;
    breg = reshape(multa1(m,et.e{ii,jj},vr),Nxy,Nzr,Ncomp);
    for kk = 1:Nzr
      for ll = 1:Ncomp
        if nnz(wi.w{ii,jj,kk,ll})
          b(wi.indl(ii,jj,kk,ll):wi.indu(ii,jj,kk,ll)) = wi.w{ii,jj,kk,ll}*breg(:,kk,ll);
        end
      end
    end
  end

end



function copy_pars_as_dual_grid
  load tmpall ;
  switch igrid;
    case 2
      x1c=x1; y1c=y1; zc=z; dzc=dz;
      xydc=xyd; dnc=dn; zrc=zr; dindc=dind;
      srcparrealc = srcparreal;
      indsrtc = indsrt;
      indinvsrtc = indinvsrt;
      sigbgc = sigbg ;
      sigtotc = sigtot;
      dsc = ds ;
      wic = wi ;
      FNxyc = FNxy;
      indbodyc = indbody;
      save fwdstg1c.mat indbodyc x1c y1c xc yc zc dzc xydc dnc f zrc dindc srcparrealc indsrtc indinvsrtc sigbgc sigtotc dsc wic FNxyc;
    case 1;
      x1f=x1; y1f=y1; zf=z; dzf=dz;
      xydf=xyd; dnf=dn; zrf=zr; dindf=dind;
      srcparrealf = srcparreal;
      indsrtf = indsrt;
      indinvsrtf = indinvsrt;
      sigbgf = sigbg ;
      sigtotf = sigtot;
      dsf = ds ;
      wif = wi ;
      FNxyf = FNxy;
      indbodyf = indbody;
      save fwdstg1.mat indbody x1 y1 x y z dz xyd dn f zr dind srcparreal indsrt indinvsrt sigbg sigtot ds wi FNxy;
      save fwdstg1f.mat indbodyf x1f y1f xf yf zf dzf  xydf dnf f zrf dindf srcparrealf indsrtf indinvsrtf sigbgf sigtotf dsf wif FNxyf;
  end;
  clear tmpall;


function [Nxf,Nyf,Nzf,Nxyf,Nxyzf,xmf,ymf,zmf,dxf] = load_fine_grid_pars(xf,yf,zf) ;
  Nxf = length(xf); Nyf = length(yf); Nzf = length(zf);
  Nxyf = Nxf*Nyf ;  Nxyzf = Nxyf*Nzf;
  [xmf,ymf,zmf] = ndgrid(xf,yf,zf);
  xmf = xmf(:); ymf = ymf(:); zmf = zmf(:);
  dxf = xf(2)-xf(1);



function show_calculation_config(solflag,N3d)
  switch solflag
    case 1
      fprintf('\n-----------\nBorn approximation\n');
    case 2
      fprintf('\n-----------\nQA approximation\n');
    case 3
      fprintf('\n-----------\nQA series approximation\n');
    case {4,5,6}
      fprintf('\n-----------\nContraction full integral solution\n');
    case {7,8}
      fprintf('\n-----------\nQL approximation\n');
    case {9,10}
      fprintf('\n-----------\nMean value LQL approximation\n');
  end
  fprintf('Size of the 3D domain : %g x %g x %g (%g cells) \n',N3d.Nx,N3d.Ny,N3d.Nz,N3d.Nxyz)
  if solflag >= 7;
    fprintf('Size of the 3D domain : %g x %g x %g (%g cells) (for final result)\n',N3d.Nxf,N3d.Nyf,N3d.Nzf,N3d.Nxyzf)
  end


function show_lql_config(Ngrp,kk,Nsrc, Nsubset);
  fprintf('Ngrp: %g\n',Ngrp);
  fprintf('kk: %g\n',kk);
  fprintf('Nsrc: %g\n',Nsrc);
  fprintf('Nsubset: %g\n',Nsubset);






function check_sig_3d(x,y,z,ds,Nxyz,sn,sig3)
  % script for sigma model with 3-D view
  % this code is originaly developed by Alex
  %
  %
  ds = [ds zeros(Nxyz,1)];
  ds0 = ds;
  ds = ds(:,1);

  if sn~=1;
    snn=sig3;
    %
    %--------------------------------------------------
    % 3D view part
    % ds    : delta sigma (vector)
    % ts    : total sigma (vector)
    % xanom : (lx,ux)
    % yanom : (ly,uy)
    % zanom : (lz,uz)
    % xcz   : x cell length (scalar)
    % ycz   : y cell length (scalar)
    % zcz   : z cell length (vector)
    % sig0  : background sigma (vector)
    % hh0   : background thickness (vector)
    %
    %
    save snn snn ;
    xanom=[min(snn(:,3)) max(snn(:,4))]
    yanom=[min(snn(:,5)) max(snn(:,6))];
    zanom=[min(snn(:,7)) max(snn(:,8))];
    xcsz=x(2)-x(1)
    ycsz=y(2)-y(1);
    %  zcsz=anodz;%mod dz>anodz
    zcsz=diff([0 z])%mod dz>anodz

    % call plotting kernel-1 by Alex (Thanks Alex!)
    isofillsz(xanom,yanom,zanom,xcsz,ycsz,zcsz,z,ds,ts);
    %
  end


function [ts] = rectbody2(x,y,z,ts,sb,sigr,lx,ux,ly,uy,lz,uz,sigi,sigt,sigf)
  %
  % 2004-0726 : separate sigbody into sigbodyr and sigbodyi
  %
  % ----------------------------------------------
  % Filling delta sigma into a rectangular body
  % ----------------------------------------------
  %
  % ts      : total sigma in each cell (being updated)
  % ds      : delta sigma in each cell
  %
  % x       : vector of the x coordinates of the mesh
  % y       : vector of the y coordinates of the mesh
  % z       : vector of the z coordinates of the mesh
  % sb      : Nxyz,1 vector of background conductivity
  % sigbody : total conductivity of the body
  % lx      : lower x boundary of the body
  % ux      : upper x boundary of the body
  % ly      : lower y boundary of the body
  % uy      : upper y boundary of the body
  % lz      : lower z boundary of the body
  % uz      : upper z boundary of the body

  [xx,yy,zz] = ndgrid(x,y,z);
  xyz = [xx(:) yy(:) zz(:)];

  ind = find(xyz(:,1) >= lx & xyz(:,1) <= ux & ...
      xyz(:,2) >= ly & xyz(:,2) <= uy & ...
      xyz(:,3) >= lz & xyz(:,3) <= uz) ;

  % 2004-0726 make complex number
  switch nargin ;
    case 12 ;
      sigbody = sigr ;
    case 13 ;
      sigbody = [sigr sigi] ;
    case 15 ;% Cole-Cole parameter
      sigbody = [sigr sigi sigt sigt] ;
  end
  ts(ind,:) = repmat(sigbody,numel(ind),1);





function sb = bgcond_1dip(h,sig,x,y,z)
% ----------------------------------------------
% Computing background conductivity in the mesh
% with complex conducitivity
% ----------------------------------------------
% h       : vector of layer thicknesses, [] for halfspace
% sig     : vector of layer conductivities
% x       : vector of the x coordinates of the mesh
% y       : vector of the y coordinates of the mesh
% z       : vector of the z coordinates of the mesh
% sb      : Nxyz,1 vector of background conductivity

h = (h(:))';
Nx=length(x); Ny=length(y); Nz=length(z);
Nxyz=Nx*Ny*Nz;
%nl = length(sig);
nl = length(h)+1;

h=[0 h inf];

zbnd=zeros(1,length(h));
zbnd(1)=h(1);
for i=2:length(h)
   zbnd(i)=zbnd(i-1)+h(i);
end

sb.real = zeros(0,1);

if isempty(sig.imag)~=1;
  sb.imag = zeros(0,1);
end
if isempty(sig.chrg)~=1;
  sb.chrg = zeros(0,1);
end
if isempty(sig.timc)~=1;
  sb.timec = zeros(0,1);
end
if isempty(sig.frqc)~=1;
  sb.freqc = zeros(0,1);
end

for i=1:nl
   ind = find((z>zbnd(i)) & (z<zbnd(i+1))) ;
   numc=length(ind)*Nx*Ny ;
   sb.real=[sb.real ; sig.real(i)*ones(numc,1)];

   if isempty(sig.imag)~=1;
     sb.imag=[sb.imag ; sig.imag(i)*ones(numc,1)];
   end
   if isempty(sig.chrg)~=1;
     sb.chrg=[sb.chrg ; sig.chrg(i)*ones(numc,1)];
   end
   if isempty(sig.timc)~=1;
     sb.timc=[sb.timc ; sig.timc(i)*ones(numc,1)];
   end
   if isempty(sig.frqc)~=1;
     sb.frqc=[sb.frqc ; sig.frqc(i)*ones(numc,1)];
   end
end




function [xc,yc,zc,dzc]=prep_input_pars_ql(x,y,z,dz,Combo)
  % preparation of coarse grid from user input fine grid
  % this subfunction is called only for SOLFLAG >= 7
  % x,y,z.dz : user input, these are used as "finer" grid
  %
  % xc,yc,zc,dzc : corase grid (double spacing)

  % compute horizontal cell length (horizontal cell length are unifrom)
  dFine.x = x(2)-x(1); dFine.y = y(2)-y(1);

  % set the number of cell combination from FINE to COARSE grid
  switch numel(Combo);
    case 1;
      nCombo.x = Combo(1) ; nCombo.y = Combo(1) ; nCombo.z = Combo(1) ;
    case 2;
      nCombo.x = Combo(1) ; nCombo.y = Combo(1) ; nCombo.z = Combo(2) ;
    case 3;
      nCombo.x = Combo(1) ; nCombo.y = Combo(2) ; nCombo.z = Combo(3) ;
    otherwise
      error('Parameter COMBO shoud has length less equal 3') ;
  end

  nFine.x = length(x) ; nFine.y = length(y); nFine.z = length(z);
  lFine.x = dFine.x * nFine.x ; lFine.y = dFine.y * nFine.y ;

    if mod(nFine.x,nCombo.x) | mod(nFine.y, nCombo.y);
    error('Horizontal coarse grid setting is not correct');
  end

  nCoarse.x = nFine.x / nCombo.x ;
  nCoarse.y = nFine.y / nCombo.y ;

  dCoarse.x = dFine.x * nCombo.x ;
  dCoarse.y = dFine.y * nCombo.y ;

  nodeCoarse.x = [x(1)-0.5*dFine.x : dCoarse.x : x(end)+0.5*dFine.x];
  nodeCoarse.y = [y(1)-0.5*dFine.y : dCoarse.y : y(end)+0.5*dFine.y];

  if ( abs(nodeCoarse.x(end) - (x(end)+0.5*dFine.x)) > 1e-8 ) & ( length(nodeCoarse.x)-1 ~= nCoarse.x) ;
      nodeCoarse.x(end+1) = x(end)+0.5*dFine.x ;
  end

  if ( abs(nodeCoarse.y(end) - (y(end)+0.5*dFine.y)) > 1e-8 ) & ( length(nodeCoarse.y)-1 ~= nCoarse.y) ;
      nodeCoarse.y(end+1) = y(end)+0.5*dFine.y ;
  end

  centerCoarse.x = nodeCoarse.x(1:end-1)+0.5*dCoarse.x ;
  centerCoarse.y = nodeCoarse.y(1:end-1)+0.5*dCoarse.y ;

  nodeFine.z = [z-0.5*dz z(end)+0.5*dz(end)] ;
  nodeCoarse.z = nodeFine.z(1:nCombo.z:end);
  if mod(nFine.z, nCombo.z)
    nodeCoarse.z = [nodeCoarse.z nodeFine.z(end)];
  end

  dCoarse.z = nodeCoarse.z(2:end) - nodeCoarse.z(1:end-1);
  nCoarse.z = numel(dCoarse.z);

  centerCoarse.z = nodeCoarse.z(1:end-1) + 0.5.* dCoarse.z;

  % set result as output arguments
  xc = centerCoarse.x;
  yc = centerCoarse.y;
  zc = centerCoarse.z;
  dzc = dCoarse.z;

  % end of prep_input_ql

function [dsc,tsc] = fillsigintem (x,y,z,hh0,sigset,sn,sig3,kcomp)
% FILLSIGINTEM.m generates SIGBODY.DAT automatically
% from user friendly input file "SBODY.DAT"
%
%  this script is based on "FILLSIG.m" by CEMI
%  and modified by Takumi Ueda
%
%  This script shoud be called by a kind of "intem3d" type MATLAB code
%  and does not require any other input files ot paramter files
%
% case: kcomp(2) == 1
%
% sig3(1,:) = [0.001 0 -50   0   0  50 100 200]
% sig3(2,:) = [0.100 0 -50   0 -50   0 100 200]
% sig3(3,:) = [1.000 0   0  50   0  50 100 200]
% sig3(4,:) = [10.00 0   0  50 -50   0 100 200]
%
% sig3(i,:) = [real imag x1 x2 y1 y2 z1 z2]
%
% case: kcomp(2) == 2
%
% sig3(1,:) = [0.1 0.01 0.5 0.2 -50   0   0  50 100 200]
% sig3(2,:) = [0.1 0.07 0.1 0.4 -50   0 -50   0 100 200]
% sig3(3,:) = [1.0 0.10 0.3 0.9   0  50   0  50 100 200]
% sig3(4,:) = [100 0.01 0.5 0.2   0  50 -50   0 100 200]
%
% sig3(i,:) = [conductivity(f=0) charge-ability time-constant freq-constant x1 x2 y1 y2 z1 z2]

Nxyz = length(x)*length(y)*length(z);

sb = bgcond_1dip(hh0,sigset,x,y,z);

switch kcomp(2)
  case 0
    dsc = zeros(Nxyz,1);
    sbc = [sb.real(:)] ;
  case 1
    dsc = zeros(Nxyz,2);
    sbc = [sb.real(:) sb.imag(:)] ;
  case 2
    dsc = zeros(Nxyz,4);
    sbc = [sb.real(:) sb.chrg(:) sb.timc(:) sb.frqc(:)] ;
end
tsc = sbc ;                       % total conductivity
switch kcomp(2)
  case 0 % Real part of conducitivity and 6 XYZ coordinates
    for i=1:sn
      [tsc] = rectbody2(x,y,z,tsc,sbc,sig3(i,1),sig3(i,2),sig3(i,3),sig3(i,4),sig3(i,5),sig3(i,6),sig3(i,7));
    end
  case 1 % Real and imaginary part of conductivity parameters and 6 XYZ coordinates
    for i=1:sn
      [tsc] = rectbody2(x,y,z,tsc,sbc,sig3(i,1),sig3(i,3),sig3(i,4),sig3(i,5),sig3(i,6),sig3(i,7),sig3(i,8),sig3(i,2));
    end
  case 2 % 4 Cole-Cole parameters and 6 XYZ coordinates
    for i=1:sn
      [tsc] = rectbody2(x,y,z,tsc,sbc,sig3(i,1),sig3(i,2),sig3(i,5),sig3(i,6),sig3(i,7),sig3(1,8),sig(i,9),sig(i,10),sig3(i,3),sig3(i,4));
    end
  otherwise
    error('kcomp(2) should be 0, 1 or 2');
end

dsc = tsc - sbc;
% ts will be returned as contents of "SIGBODY.DAT"
% ds will not be returned but used to show grid model below
%


function [err] = prep_sigbody(hh0,sigset,kcomp,x,y,z,dz,xc,yc,zc,dzc)
exist_sigbody = exist('sigbody_tmp.dat','file') ;

switch nargin
 case 7;
  % single grid mode

  % step 1 : check "sigbody.dat"
  if exist_sigbody ~= 2 ;
    error('sigbody.dat is not exist ');

  elseif exist_sigbody == 2 ;
    load sigbody_tmp.dat ;
    ts = sigbody_tmp ;
    sizets = size(ts) ;
    if sizets(2) >= 7 ;
      %mosvcp('sigbody.dat','sigbody.dat.user');
      sn = sizets(1) ;
      [ds,ts] = fillsigintem(x,y,z,hh0,sigset,sn,ts,kcomp);
      save sigbody_tmp.dat ts -ascii;
    elseif sizets(1) == 1 | sizets(2) <= 4;
      ;
    end


    %elseif exist('sbody.dat','file');
    %  mosvcp('sbody.dat','sbody.m');
    %  sbody; % load "sbody.m"
    %  [ds,ts] = fillsigintem(x,y,z,hh0,sigset,sn,sig3,kcomp);
    %  save sigbody.dat ts -ascii;
  else
    err = 1;
    error('Error with sigbody.dat  ')
  end

 case 11
  % dual grid mode

  if exist_sigbody ~= 2 ;
    error('sigbody.dat is not exist ');
  elseif exist_sigbody == 2 ;
    load sigbody_tmp.dat ;
    ts = sigbody_tmp ;
    %pause

    sizets = size(ts);
    mosvcp('sigbody_tmp.dat','sigbodyf.dat');

    if sizets(1) == 1 & sizets(2) <= 4;
      mosvcp('sigbody_tmp.dat','sigbodyc.dat');
    else
      if sizets(2) <= 4 ;
        ; % nothing to do here
      elseif sizets(2) >= 7 ;
        %mosvcp('sigbody.dat','sigbody.dat.user');
        sn = sizets(1) ;
        [ds,ts] = fillsigintem(x,y,z,hh0,sigset,sn,ts,kcomp) ;
        save sigbodyf.dat ts -ascii;
      else
        error('sigbody.dat is not correct ')
      end
      load sigbodyf.dat;
      ts = sigbodyf ;
      % prep_coarse_sigbody
      [err] = prep_coarse_sigbody(ts,x,y,z,dz,xc,yc,zc,dzc);
    end
  else
    error('sigbody.dat nor sbody.dat ')
  end


 otherwise
  error('input argments are not correct')
end

err = 0 ;


function [err] = prep_coarse_sigbody(ts,x,y,z,dz,xc,yc,zc,dzc);
typeSigma = size(ts,2) ;

switch typeSigma
 case 1
  [err] = gen_coarse_grid(typeSigma,ts,x,y,z,dz,xc,yc,zc,dzc);
 otherwise
  err = 1;
  error('At this moment, only real conductivity is supported')
end

function [err] = gen_coarse_grid(typeSigma,ts,x,y,z,dz,xc,yc,zc,dzc);

sizeFine.x = numel(x) ;
sizeFine.y = numel(y) ;
sizeFine.z = numel(z) ;
sizeCoarse.x = numel(xc) ;
sizeCoarse.y = numel(yc) ;
sizeCoarse.z = numel(zc) ;

sizeCoarse.xyz = [sizeCoarse.x, sizeCoarse.y, sizeCoarse.z];

% 3D array of sigma for the fine grid
tsFine = reshape(ts,sizeFine.x , sizeFine.y, sizeFine.z);

nCombo.x = sizeFine.x ./ sizeCoarse.x ;
nCombo.y = sizeFine.y ./ sizeCoarse.y ;

if mod(sizeFine.z, sizeCoarse.z) == 0
  nCombo.z = sizeFine.z ./ sizeCoarse.z ;
else
  nCombo.z = (sizeFine.z + (sizeCoarse.z - mod(sizeFine.z, sizeCoarse.z))) ./ sizeCoarse.z ;
  %pause
end

if nCombo.x == nCombo.y
  nCombo.xy = nCombo.x ;
  if nCombo.xy == nCombo.z
    nCombo.xyz = nCombo.xy;
  end
end

tsCoarse.x   = zeros(sizeCoarse.x, sizeFine.y, sizeFine.z);
tsCoarse.xy  = zeros(sizeCoarse.x, sizeCoarse.y, sizeFine.z);
tsCoarse.xyz = zeros(sizeCoarse.xyz);

% combine fine grid for X direction
for iCombo = 1 : nCombo.x ;
  tsCoarse.x = tsCoarse.x + tsFine(iCombo : nCombo.x : end, :, :);
end

tsCoarse.x = tsCoarse.x ./ nCombo.x ; % take average


% combine fine grid for Y direction
for iCombo = 1 : nCombo.y ;
  tsCoarse.xy = tsCoarse.xy + tsCoarse.x(:, iCombo : nCombo.y : end, :, :);
end

tsCoarse.xy = tsCoarse.xy ./ nCombo.y ; % take average

% combine fine grid for Z direction
% Z direction is different from X and Y directions
irrCoarsez = mod(sizeFine.z, nCombo.z) ; % check coarse grid

% preparation for cell length for Z direction
% When we combine fine cells to coarse grid, we need to compute averaged value with simple length weights.
if irrCoarsez;

  endCombo.z = nCombo.z * (sizeCoarse.z-1) ;
  dzWeight = reshape(dz(1:endCombo.z), nCombo.z, sizeCoarse.z -1);
  dzWeight = dzWeight ./ repmat(sum(dzWeight), nCombo.z ,1); % generate weight
  for iCombo = 1 : nCombo.z ;
    dzWeight2d = repmat(dzWeight(iCombo,:), sizeCoarse.x * sizeCoarse.y, 1);
    tsCoarse.xyz(:,:,1:end-1) = tsCoarse.xyz(:,:,1:end-1) + reshape(dzWeight2d(:), sizeCoarse.x, sizeCoarse.y, sizeCoarse.z-1) .* tsCoarse.xy(:, : , iCombo:nCombo.z:endCombo.z ) ;
  end
  % if there are extra fine cells, we combine all extra cells as the last cell for the coarse grid.
  % generate cell length weight for the last coarse cell
  dzWeightLast = dz(endCombo.z+1 : end ) ./ dzc(end) ;
  %sum(dz(endCombo.z+1:end));
  %dzWeightLast2d  = repmat(dzWeightLast, sizeCoarse.x * sizeCoarse.y,1);

  for iCombo = 1 : irrCoarsez;
    tsCoarse.xyz(:,:,end) = tsCoarse.xyz(:,:,end) + dzWeightLast(iCombo).* tsCoarse.xy(: , : , endCombo.z + iCombo);
  end

else ; % for uniform cell length in Z direction

  endCombo.z = sizeFine.z ;
  endCombo.z ;
  dzWeight = reshape(dz, nCombo.z, sizeCoarse.z) ;
  dzWeight = dzWeight ./ repmat(sum(dzWeight), nCombo.z ,1) ;% generate weight

  for iCombo = 1 : nCombo.z ;
    dzWeight2d = repmat(dzWeight(iCombo,:), sizeCoarse.x * sizeCoarse.y, 1);
    tsCoarse.xyz = tsCoarse.xyz + reshape(dzWeight2d(:), sizeCoarse.xyz).* tsCoarse.xy(:, : , iCombo:nCombo.z:endCombo.z ) ;
  end

end
tsc = tsCoarse.xyz(:) ;
save sigbodyc.dat tsc -ascii;

err = 0 ;



function [lamdaf] = lamdafine(lamda,Nxyz,Nx,Ny,Nz,x,y,z,xf,yf,zf,swint)
% interpolate Lamda to finer grid
% 2004 Fall   : Initial version (dual grid only)
% 2005 Spring : Arbitrary grids
% 2006 Fall   : Begin to modify but not yet finished

%
%  This is draft version
%

% check input argument
if nargin == 11 ;swint =0;end
% swint : 0 < just re-distribute
%         1 < interpolation

% step.1 : reshape and purge x, y and z
lamda3.xyz = reshape(lamda,Nxyz,3);
lamda3.x = reshape(lamda3.xyz(:,1),Nx,Ny,Nz);
lamda3.y = reshape(lamda3.xyz(:,2),Nx,Ny,Nz);
lamda3.z = reshape(lamda3.xyz(:,3),Nx,Ny,Nz);
%
% prepare 3D (x, y, z) meshgrids
[xf3,yf3,zf3] = meshgrid(yf,xf,zf);
[xc3,yc3,zc3] = meshgrid(y,x,z);

% size of FINE & COARSE grids
sizeFine.x = numel(xf);  sizeFine.y = numel(yf);  sizeFine.z = numel(zf);
sizeCoarse.x = numel(x);  sizeCoarse.y = numel(y);  sizeCoarse.z = numel(z);

combo.x = sizeFine.x / sizeCoarse.x ;
combo.y = sizeFine.y / sizeCoarse.y ;
flagcomboz = mod(sizeFine.z,sizeCoarse.z);

sizeFine.xyz = [sizeFine.x, sizeFine.y, sizeFine.z];
sizeCoarse.xyz = [sizeCoarse.x, sizeCoarse.y, sizeCoarse.z];

switch swint
 case 0
  %
  % swint = 0 : no interpolation, but re-distribute
  %
  Nxf = length(xf); Nyf = length(yf); Nzf = length(zf);

  for i = 1:3 ;
    switch i
     case 1;
      insparse=lamda3.x;
     case 2;
      insparse=lamda3.y;
     case 3;
      insparse=lamda3.z;
    end
    outfine = zeros(Nxf, Nyf, Nzf);

    if flagcomboz == 0 ;
      combo.z = sizeFine.z / sizeCoarse.z ;
      for j = 1 : sizeCoarse.z ;
        inptz=insparse(:,:,j);
        t1 = inptz(:);
        t1 = repmat(t1.',combo.x,1);
        t1 = reshape(t1,Nxf,Ny);
        t1 = repmat(t1,combo.y,1);
        outfine(:,:,combo.z*(j-1)+1) = reshape(t1,Nxf,Nyf);
        outfine(:,:,combo.z*j) = outfine(:,:,combo.z*(j-1)+1) ;
      end
    else
      combo.z = (sizeFine.z + (sizeCoarse.z - mod(sizeFine.z, sizeCoarse.z))) ./ sizeCoarse.z;
      for j = 1 : sizeCoarse.z - 1 ;
        inptz=insparse(:,:,j);
        t1 = inptz(:);
        t1 = repmat(t1.',combo.x,1);
        t1 = reshape(t1,Nxf,Ny);
        t1 = repmat(t1,combo.y,1);
        outfine(:,:,combo.z*(j-1)+1) = reshape(t1,Nxf,Nyf);
        outfine(:,:,combo.z*j) = outfine(:,:,combo.z*(j-1)+1) ;
      end
      inptz=insparse(:,:,end);
      t1 = inptz(:);
      t1 = repmat(t1.',combo.x,1);
      t1 = reshape(t1,Nxf,Ny);
      t1 = repmat(t1,combo.y,1);
      for j = 1 : flagcomboz
        outfine(:,:,end-flagcomboz+j) = reshape(t1,Nxf,Nyf);
      end
    end

    switch i
     case 1;
      lamda3f.x = outfine;
     case 2;
      lamda3f.y = outfine;
     case 3;
      lamda3f.z = outfine;
    end

  end ; % end of for i=1:3 (x, y and z)

 case 1
  % interpolate to finer grid
  % MATLAB buit-in function "interp3" with 'spline' method
  [lamda3f.x] = interp3(xc3,yc3,zc3,lamda3.x,xf3,yf3,zf3,'spline');
  [lamda3f.y] = interp3(xc3,yc3,zc3,lamda3.y,xf3,yf3,zf3,'spline');
  [lamda3f.z] = interp3(xc3,yc3,zc3,lamda3.z,xf3,yf3,zf3,'spline');

  % replace edge value to avoid SPLINE extrapolation
  % English : incorrect ; outerpolation
%             correct ; extrapolation

  [lamda3f.x] = replacedge(lamda3f.x);
  [lamda3f.y] = replacedge(lamda3f.y);
  [lamda3f.z] = replacedge(lamda3f.z);
end

%reshape and merge to one column vector
%lamdaf.x = reshape(lamda3f.x,Nxyzf,1);
%lamdaf.y = reshape(lamda3f.y,Nxyzf,1);
%lamdaf.z = reshape(lamda3f.z,Nxyzf,1);
lamdaf.x = lamda3f.x(:);
lamdaf.y = lamda3f.y(:);
lamdaf.z = lamda3f.z(:);
lamdaf = [lamdaf.x ; lamdaf.y ; lamdaf.z];







function [] = mosvcp(sourcef,destinationf)
%MOSVCP  What is this
%   MOSVCP(A,B) is an altanative "copy file" function for Unix and Win32.
%
%   See also CHKMLVER

if nargin ~= 2
  error('Source and destination file/dir name should be given');
end
[mversion] = chkmlver ;

%  system(['cp -r ' sourcef ' ' destinationf])

switch mversion;
 case {6.5,7};
  if isunix
     system(['cp -r ' sourcef ' ' destinationf]);
  elseif ispc
      copyfile(sourcef,destinationf);
  end
 case {6,5}
  if isunix;
    copyfile(sourcef,destinationf);
  else
    error('This version is only for MATLAB 7/6.5 (Win32/Linux) or 6.0/5.0 (Linux)');
  end
end

return

function [mversion]=chkmlver;
%
% Last modified: 2006/03/08 11:47:57 MST
%
v=version;
vm = sscanf(v,'%f');
if vm(1)==6.5 ;%& vm(2)==0; [2006-03-08] for ver. 6.5.1
    mversion=6.5;
elseif vm(1)>=7; %[2006-01-18] for r14sp3
    mversion=7;
elseif vm(1)==6 & vm(2)==0;
    mversion=6.0;
elseif vm(1)==5;
    mversion=5;
else
    disp('Your MATLAB version may not be supported by this code');
    disp('If you continue calculation, you may have trouble...') ;
    mversion=7 ; % [2006-03-08]
end

return
