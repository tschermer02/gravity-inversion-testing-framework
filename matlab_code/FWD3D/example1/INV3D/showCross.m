close all
load(['Final' filesep 'invres.mat'])
m1=mPars{1}.sigT;
m2=mPars{4}.sigT;
semilogx(m2,m1,'.')
xlabel('Conductivity')
ylabel('Density')
