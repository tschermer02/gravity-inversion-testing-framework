% GREEN3D electromagnetic green's tensor matlab library
%
% [e,h]=green3d(f,hl,sl,al,xr,yr,zr,par);if par = -1 or 1 <= par(1) <= 8
%
%  e = green3d(f,hl,sl,al,xr,yr,zr,par) if par(1) = -2
%
%  h = green3d(f,hl,sl,al,xr,yr,zr,par) if par(1) = -3
%
% Computation of Green's tensors and primary fields.
% Input and output arguments:
%
% f    : frequency
%
% hl   : vector of layer thicknesses
%
% sl   : vector of complex layer conductivities
%
% al   : vector of layer anisotropies
%
% xr   :
% yr   : receiver x,y,z coordinates
% zr   :
%
% par(1):     = -3...-1 - Green's tensor computation
%             = 1...8   - primary field computation
%
% par(2:end)  : if -3 <= par(1) <= -1
%               vector if six cell parameters like in cell.dat
%               See green library documentation
%               if 1 <= par(1) <= 8
%               vector of source parameters like in the second
%               row of source.dat. See green library documentation
%
% e,h  : if par(1) = -3 <= par(1) <= -1
%          e and h are [size(xr) 3 3] blocks of green tensor elements
%        if 1 <= par(1) <= 8
%          e and h are [size(xr) 3]   block of primary fields
%
% For details, see the MATLAB source code green3d.m or the
% documentation file.


%   Green3d is a MATLAB function which can be called as
%
%        [e,h]=green3d(f,hl,sl,al,xr,yr,zr,par);
%
%   INPUT PARAMETERS:
%
%    f   : frequency (real scalar)
%
%    hl  : real vector of layer thicknesses. The first element corresponds to
%          the thickness of the uppermost layer in the earth, the last
%          element is the thickness of the lowermost bounded layer.
%
%    sl  : complex vector of layer conductivities.
%
%    al  : real vector of layer anisotropy coefficients.
%
%    xr  : real block of x coordinates of the receivers with arbitrary dimensions.
%
%    yr  : real block of y coordinates of the receivers with arbitrary dimensions.
%
%    zr  : real block of z coordinates of the receivers with arbitrary dimensions.
%
%          The dimensions of xr, yr and zr must match.
%
%    par : real vector of cell or source parameters.
%          The first element of par is a flag determining whether Green's tensor
%          integrals or normal fields are to be computed.
%          The possible structures for par are the following:
%
%       par(1) = -1    -> volume integrals of the electric and magnetic Green's
%                         tensors are computed.
%
%       par(1) = -2    -> volume integrals of the electric Green's
%                         tensors are computed only.
%
%       par(1) = -3    -> volume integrals of the magnetic Green's
%                         tensors are computed only.
%
%                         If par(1) = -1 ... -3, par has a length of 7.
%                         The meaning of the 2-7. elements of par is
%
%          par(2)      -> dimension of the cell in x direction
%
%          par(3)      -> dimension of the cell in y direction
%
%          par(4)      -> dimension of the cell in z direction
%
%          par(5)      -> x coordinate of the cell's center
%
%          par(6)      -> y coordinate of the cell's center
%
%          par(7)      -> z coordinate of the cell's center
%
%       par(1) = 1    ->  electromagnetic field due to a plane wave is computed.
%                         In this case, par has no additional elements, because
%                         no source parameters are necessary.
%
%       par(1) = 2    -> electromagnetic field due to a vertical electric bipole is
%                        computed. In this case, par has a length of 8. The meaning
%                         of the 2-8. elements of par is
%
%          par(2)      -> current strength in the source [Amper]
%
%          par(3)      -> x coordinate of the A electrode
%
%          par(4)      -> y coordinate of the A electrode
%
%          par(5)      -> z coordinate of the A electrode
%
%          par(6)      -> x coordinate of the B electrode
%
%          par(7)      -> y coordinate of the B electrode
%
%          par(8)      -> z coordinate of the B electrode
%
%                 Only the z coordinates of the A and B electrodes can be different
%
%
%       par(1) = 3    -> electromagnetic field due to a horizontal electric bipole is
%                        computed. In this case, par has a length of 8. The meaning
%                         of the 2-8. elements of par is
%
%          par(2)      -> current strength in the source [Amper]
%
%          par(3)      -> x coordinate of the A electrode
%
%          par(4)      -> y coordinate of the A electrode
%
%          par(5)      -> z coordinate of the A electrode
%
%          par(6)      -> x coordinate of the B electrode
%
%          par(7)      -> y coordinate of the B electrode
%
%          par(8)      -> z coordinate of the B electrode
%
%         Only one of the x or y coordinates of the A and B electrodes can be different
%
%
%       par(1) = 4    -> electromagnetic field due to a horizontal rectangular loop is
%                        computed. The loop is specified by four current electrodes,
%                        A, B, C and D. In this case, par has a length of 14. The meaning
%                        of the 2-14. elements of par is
%
%          par(2)      -> current strength in the source [Amper]
%
%          par(3)      -> x coordinate of the A electrode
%
%          par(4)      -> y coordinate of the A electrode
%
%          par(5)      -> z coordinate of the A electrode
%
%          par(6)      -> x coordinate of the B electrode
%
%          par(7)      -> y coordinate of the B electrode
%
%          par(8)      -> z coordinate of the B electrode
%
%          par(9)      -> x coordinate of the C electrode
%
%          par(10)     -> y coordinate of the C electrode
%
%          par(11)     -> z coordinate of the C electrode
%
%          par(12)     -> x coordinate of the D electrode
%
%          par(13)     -> y coordinate of the D electrode
%
%          par(14)     -> z coordinate of the D electrode
%
%
%            The coordinates of the loop have to be specified such that the
%            sides are parallel to the x or y axes.
%
%
%       par(1) = 5    -> electromagnetic field due to a horizontal circular loop
%                        computed. In this case, par has a length of 6. The meaning
%                         of the 2-6. elements of par is
%
%          par(2)      -> current strength in the source [Amper]
%
%          par(3)      -> x coordinate of the center of the loop
%
%          par(4)      -> y coordinate of the center of the loop
%
%          par(5)      -> z coordinate of the center of the loop
%
%          par(6)      -> The radius of the loop [meter]
%
%       par(1) = 6    -> electromagnetic field due to a magnetic dipole oriented in
%                        x direction is computed. In this case, par has a length of 4.
%                        The meaning of the 2-4. elements of par is
%
%          par(2)      -> x coordinate of the dipole
%
%          par(3)      -> y coordinate of the dipole
%
%          par(4)      -> z coordinate of the dipole
%
%       par(1) = 7    -> electromagnetic field due to a magnetic dipole oriented in
%                        y direction is computed. In this case, par has a length of 4.
%                        The meaning of the 2-4. elements of par is
%
%          par(2)      -> x coordinate of the dipole
%
%          par(3)      -> y coordinate of the dipole
%
%          par(4)      -> z coordinate of the dipole
%
%       par(1) = 8    -> electromagnetic field due to a magnetic dipole oriented in
%                        z direction is computed. In this case, par has a length of 4.
%                        The meaning of the 2-4. elements of par is
%
%          par(2)      -> x coordinate of the dipole
%
%          par(3)      -> y coordinate of the dipole
%
%          par(4)      -> z coordinate of the dipole
%
%       par(1) = 9    -> electromagnetic field due to a electric dipole (unit length) oriented in
%                        arbitrary 3D direction is computed. In this case, par has a length of 6.
%                        The meaning of the 2-6 elements of par is
%
%          par(2)      -> x coordinate of the dipole
%
%          par(3)      -> y coordinate of the dipole
%
%          par(4)      -> z coordinate of the dipole
%
%          par(5)      -> the angle of the dipole from x-axis
%          (counterclockwise) in degree (-180 < par(5) <= 180)
%
%          par(6)      -> the angle of the dipole from xy-plane (downward)
%          in degree (-90 <= par(6) <=90)
%
%       par(1) = 10    -> electromagnetic field due to a magnetic dipole oriented in
%                        arbitrary 3D direction is computed. In this case, par has a length of 6.
%                        The meaning of the 2-6 elements of par is
%
%          par(2)      -> x coordinate of the dipole
%
%          par(3)      -> y coordinate of the dipole
%
%          par(4)      -> z coordinate of the dipole
%
%          par(5)      -> the angle of the dipole from x-axis (counterclockwise) in degree
%
%          par(6)      -> the angle of the dipole from xy-plane (downward) in degree
%
%   OUTPUT PARAMETERS:
%
%    e   : if par(1) = -1 or -2, e is a [sizeof(xr),3 3] array of the
%          volume integral of the electric Green's tensor.
%
%          if par(1) = 1..8, then e is a [sizeof(xr),3] array of the
%          electric fields due to the corresponding source.
%
%    h   : if par(1) = -1 or -3, h is a [sizeof(xr),3 3] array of the
%          volume integral of the magnetic Green's tensor.
%
%          if par(1) = 1..8, then h is a [sizeof(xr),3] array of the
%          magnetic fields due to the corresponding source.
%
%          if par(1) = -2, only the electric Green's tensor returns
%          if par(1) = -2, only the magnetic Green's tensor returns
%
%  -----------------------------------------------------------------------
%                             EXAMPLES
%  -----------------------------------------------------------------------
%
%  Consider a three layered medium with layer resistivities of 40, 50 and 25
%  ohm-m, respectively. The thickness of the first layer is 20 m, while the second
%  layer is 35 m thick. The anisotropy coefficient is 1 in all layers, e.g. no
%  anisotropy is considered. The frequency is 150 Hz in every case.
%  Let us use an array of receivers located at the nodes of a three dimensional grid
%  specified by three vectors x = [1 2 3 4 5 6], y = [-2 -1 0 1 2] and z = [6 7 8 9].
%
%
%  First, we compute the volume integral of the electric and magnetic Green's tensor
%  over a rectangular cell of size of 5, 7 and 9 meters in x, y and z directions,
%  respectively. The coordinates of the center of the cell are x = 0, y = 2 and
%  z = 45 meters.
%
%  f = 150;               % frequency
%  hl = [20 35];          % thicknesses
%  sl = [0.025 0.02 0.04] % layer conductivities. No imaginary parts are considered.
%  al = [1 1 1]           % anisotropies
%  [xr,yr,zr] = ndgrid([1 2 3 4 5 6],[-2 -1 0 1 2],[6 7 8 9]); % receiver coordinates
%  par = [-1 5 7 9 0 2 45]; % par(1) = -1, the rest describes the cell
%  [e,h]=green3d(f,hl,sl,al,xr,yr,zr,par);
%  size(e)
%
%
%  Second, let us compute the electric and magnetic fields of a vertical electric
%  bipole with the following electrode locations:
%        electrode    x     y     z   Current strength (A)
%            A        0     3     35         1
%            B        0     3     45
%  The only parameter to be modified is "par". So, by
%
%  par = [2 1 0 3 35 0 3 45]; % par(1) = 2, the rest describes the source.
%  [e,h]=green3d(f,hl,sl,al,xr,yr,zr,par);
%  size(e)
%
%  we obtain the EM field components of the bipole source.
%
%  Note the sizes of the output arrays retain the sizes of the receiver coordinate
%  arrays. This can be useful for construction of multidimensional modeling
%  and inversion codes.
%
%  In the testing routine "tst.m" you can find an example for each source and the
%  Green's tensor integration as well.
%

function [e,h]=green3d(f,hl,sl,al,xr,yr,zr,par)

%--------------------------------------------------------------
% Warnings and error messages for incorrect input and output
%--------------------------------------------------------------

if ~( ((par(1)>=-3) && (par(1)<=-1)) || ((par(1)>=1) && (par(1)<=10)) )
  error('par(1) has an invalid value');
end

if ~isequal(size(xr),size(yr),size(zr))
  error('The dimensions of xr, yr and zr must agree');
end

if ((nargout < 1) || (nargout > 2)) 
    error('The number of outputs has to be 1 or 2'); 
end

if (par(1)==-1)
  if(nargout ~= 2) 
      error('The number of outputs has to be 2'); 
  end
end

if (par(1)==-2)
  if (nargout ~= 1) 
      warning('Only the electric Greens tensor returns');
  end
end

if (par(1)==-3)
  if (nargout ~= 1) 
      warning('Only the magnetic Greens tensor returns');
  end
end

%--------------------------------------------------------------
% preparing input vectors for calling grint or normal
%--------------------------------------------------------------
rec = [xr(:) yr(:) zr(:)];
hl = hl(:);
recond = real(sl(:)); imcond = imag(sl(:));
anis = al(:);

%--------------------------------------------------------------
% compute volume integrated electric and/or magnetic Greens tensor
%--------------------------------------------------------------
if(par(1)<=-1)
  cell = par(:);
  if (length(cell) ~= 7) 
      error('Number of cell parameters should be 6'); 
  end
  if length(find(cell(2:4)<=0)) > 0
    error('Cell dimension is nonpositive');
  end
  clear grint;
  [e,h] = grint(rec,cell,f,hl,recond,imcond,anis);
  grdim = [size(xr) 3 3];
end

%--------------------------------------------------------------
% normal field
%--------------------------------------------------------------
if ((par(1)>=1) && (par(1)<=8))
 src=par(:);
 clear normal;
 [e,h] = normal(rec,src,f,hl,recond,imcond,anis);
 grdim = [size(xr) 3];
 if(par(1)==1)
  e=reshape(e,length(e)/6,3,2);
  h=reshape(h,length(h)/6,3,2);
  e(:,2,:)=e(:,1,:);
  h(:,2,:)=h(:,1,:);
  h(:,1,:)=-h(:,1,:);
 end
end
if (par(1)==9) % Modified for arbitrary oriented electric dipole (bipole with unit length)
    clear normal;
    e1=[];e2=[];e3=[];
    h1=[];h2=[];h3=[];
    xx=par(2);
    yy=par(3);
    zz=par(4);
    alp=par(5)*pi/180;
    beta=par(6)*pi/180;
    src1=[3 1 xx-cos(beta)*cos(alp)/2 yy zz xx+cos(beta)*cos(alp)/2 yy zz];
    src2=[3 1 xx yy-cos(beta)*sin(alp)/2 zz xx yy+cos(beta)*sin(alp)/2 zz];
    src3=[2 1 xx yy zz-sin(beta)/2 xx yy zz+sin(beta)/2];
    src1=src1(:);
    src2=src2(:);
    src3=src3(:);
    if par(6)==0
        if (par(5)==90)||(par(5)==-90)
            [e,h]=normal(rec,src2,f,hl,recond,imcond,anis);
            e=e*sin(alp);h=h*sin(alp);
        elseif (par(5)==0)||(par(5)==180)
            [e,h]=normal(rec,src1,f,hl,recond,imcond,anis);
            e=e*cos(alp);h=h*cos(alp);
        else
            [e1,h1] = normal(rec,src1,f,hl,recond,imcond,anis);
            [e2,h2] = normal(rec,src2,f,hl,recond,imcond,anis);
            e=e1+e2;
            h=h1+h2;
        end
    elseif abs(par(6))<90
        if (par(5)==90)||(par(5)==-90)
            [e2,h2] = normal(rec,src2,f,hl,recond,imcond,anis);
            [e3,h3] = normal(rec,src3,f,hl,recond,imcond,anis);
            e=e2+e3;
            h=h2+h3;
        elseif (par(5)==0)||(par(5)==180)
            [e1,h1] = normal(rec,src1,f,hl,recond,imcond,anis);
            [e3,h3] = normal(rec,src3,f,hl,recond,imcond,anis);
            e=e1+e3;
            h=h1+h3;
        else
            [e1,h1] = normal(rec,src1,f,hl,recond,imcond,anis);
            [e2,h2] = normal(rec,src2,f,hl,recond,imcond,anis);
            [e3,h3] = normal(rec,src3,f,hl,recond,imcond,anis);
            e=e1+e2+e3;
            h=h1+h2+h3;
        end
    else
        [e,h]=normal(rec,src3,f,hl,recond,imcond,anis);
    end
    grdim = [size(xr) 3];
end
if (par(1)==10) % Modified for arbitrary oriented magnetic dipole
    clear normal;
    xx=par(2);
    yy=par(3);
    zz=par(4);
    alp=par(5)*pi/180;
    beta=par(6)*pi/180;
    src1=[6 xx yy zz];
    src2=[7 xx yy zz];
    src3=[8 xx yy zz];
    src1=src1(:);
    src2=src2(:);
    src3=src3(:);
    if par(6)==0
        if (par(5)==90)||(par(5)==-90)
            [e,h]=normal(rec,src2,f,hl,recond,imcond,anis);
            e=e*(sin(alp)+1-1);h=h*(sin(alp)+1-1);
        elseif (par(5)==0)||(par(5)==180)
            [e,h]=normal(rec,src1,f,hl,recond,imcond,anis);
            e=e*(cos(alp)+1-1);h=h*(cos(alp)+1-1);
        else
            [e1,h1] = normal(rec,src1,f,hl,recond,imcond,anis);
            [e2,h2] = normal(rec,src2,f,hl,recond,imcond,anis);
            e=(cos(beta)+1-1)*(cos(alp)+1-1)*e1+(cos(beta)+1-1)*(sin(alp)+1-1)*e2;
            h=(cos(beta)+1-1)*(cos(alp)+1-1)*h1+(cos(beta)+1-1)*(sin(alp)+1-1)*h2;
        end
    elseif abs(par(6))<90
        if (par(5)==90)||(par(5)==-90)
            [e2,h2] = normal(rec,src2,f,hl,recond,imcond,anis);
            [e3,h3] = normal(rec,src3,f,hl,recond,imcond,anis);
            e=(cos(beta)+1-1)*(sin(alp)+1-1)*e2+(sin(beta)+1-1)*e3;
            h=(cos(beta)+1-1)*(sin(alp)+1-1)*h2+(sin(beta)+1-1)*h3;
        elseif (par(5)==0)||(par(5)==180)
            [e1,h1] = normal(rec,src1,f,hl,recond,imcond,anis);
            clear normal
            [e3,h3] = normal(rec,src3,f,hl,recond,imcond,anis);
            e=(cos(beta)+1-1)*(cos(alp)+1-1)*e1+(sin(beta)+1-1)*e3;
            h=(cos(beta)+1-1)*(cos(alp)+1-1)*h1+(sin(beta)+1-1)*h3;
        else
            [e1,h1] = normal(rec,src1,f,hl,recond,imcond,anis);
            [e2,h2] = normal(rec,src2,f,hl,recond,imcond,anis);
            [e3,h3] = normal(rec,src3,f,hl,recond,imcond,anis);
            e=(cos(beta)+1-1)*(cos(alp)+1-1)*e1+(cos(beta)+1-1)*(sin(alp)+1-1)*e2+(sin(beta)+1-1)*e3;
            h=(cos(beta)+1-1)*(cos(alp)+1-1)*h1+(cos(beta)+1-1)*(sin(alp)+1-1)*h2+(sin(beta)+1-1)*h3;
        end
    else
        [e,h]=normal(rec,src3,f,hl,recond,imcond,anis);
    end
    grdim = [size(xr) 3];
end

%--------------------------------------------------------------
% reshape the output according to receivers array dimensions
%--------------------------------------------------------------
e = squeeze( reshape(reshape(e,prod(grdim),2)*[1;i],grdim) );
h = squeeze( reshape(reshape(h,prod(grdim),2)*[1;i],grdim) );


if (par(1)==-2)
  h = [];
end

if (par(1)==-3)
  e = h;
  h = [];
end

return
