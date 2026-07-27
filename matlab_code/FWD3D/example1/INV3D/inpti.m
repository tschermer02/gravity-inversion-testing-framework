bckg{1}=0;%1-density
thick{1}=[];

bckg{2}=0;%2-magnetization
thick{2}=[];

bckg{3}=[0 0 0];%3-magnetization vector
thick{3}=[];
Bo=50000;
Ao=0;
Io=75;
Do=25;

bckg{4}=1e-3;%4-conductivity
thick{4}=[];

invBounds=[-300 300 -300 300 0 200];
xySize=[25 25];
dz=10;

dataFile{1}='obsData1.dat';
dataFile{2}='obsData2.dat';
dataFile{3}='obsData3.dat';
dataFile{4}='obsData4.dat';
Nit=300;

WmCoef={1,0.06,0.06,0.01};%property weighting coeff
alpIni={1e-6,1e-6,1e-6,1e-6};%min-norm stab coeff
alpGIni={0,0,0,0};%max-smooth stab coeff
betIni=1e-3;%Gramian coeff
betGIni=0;%gradient-type Gramian coeff

compFlag=[1 0 0 1];
dispLims=invBounds;
dispXY=xySize;
dispDz=dz;
lims3D{1}=[0.8 1.2];
barLims{1}=[0 1];
lims3D{2}=[0.05 0.07];
barLims{2}=[0 0.06];
lims3D{3}=[0.05 0.07];
barLims{3}=[-0.06 0.06];
lims3D{4}=[50 150];
barLims{4}=[1 1e3];
xs=0;
ys=[];
zs=75;
