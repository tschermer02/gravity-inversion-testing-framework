bckg{1}=0;%1-density
thick{1}=[];
anomBounds{1}=[-100 100 -100 100 50 100];
xySize{1}=[10 10];
dz{1}=10;
anom{1}=1;

bckg{2}=0;%2-magnetization
thick{2}=[];
anomBounds{2}=[-100 100 -100 100 50 100];
xySize{2}=[10 10];
dz{2}=10;
anom{2}=0.06;
Bo=50000;
Ao=0;
Io=75;
Do=25;

bckg{3}=[0 0 0];%3-magnetization vector
thick{3}=[];
anomBounds{3}=[-100 100 -100 100 50 100];
xySize{3}=[10 10];
dz{3}=10;
anom{3}=[1 0 1]*0.06/sqrt(2);

bckg{4}=1/1000;%4-conductivity
thick{4}=[];
anomBounds{4}=[-100 100 -100 100 50 100];
xySize{4}=[10 10];
dz{4}=10;
anom{4}=1/100;

rx{1}=repmat(-250:50:250,1,11)';
ry{1}=repmat(-250:50:250,11,1);ry{1}=ry{1}(:);
rz{1}=zeros(size(rx{1}))-30;
rc{1}=4:10;%Gt,Gx,Gy,Gz,Gxx,Gyy,Gzz,Gxy,Gzx,Gzy,Gd

rx{2}=repmat(-250:50:250,1,11)';
ry{2}=repmat(-250:50:250,11,1);ry{2}=ry{2}(:);
rz{2}=zeros(size(rx{2}))-30;
rc{2}=1:4;%TMI,Hx,Hy,Hz

rx{3}=repmat(-250:50:250,1,11)';
ry{3}=repmat(-250:50:250,11,1);ry{3}=ry{3}(:);
rz{3}=zeros(size(rx{3}))-30;
rc{3}=1:4;%TMI,Hx,Hy,Hz

rx{4}=repmat(-250:50:250,1,11)';
ry{4}=repmat(-250:50:250,11,1); ry{4}=ry{4}(:);
rz{4}=zeros(size(rx{4}))-30;
rc{4}=1:4;%(DIGHEM)CP56000,CP7200,CP900,CX5500,CX900

compFlag=[1 0 0 1];
dispLims=[-300 300 -300 300 0 200];
dispXY=[10 10];
dispDz=10;
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
