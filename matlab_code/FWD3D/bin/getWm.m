function Wm=getWm(Frechet)
Wm=dot(Frechet,Frechet,1).^0.25;
Wm=Wm(:);
maxWm=max(Wm);
Wm=Wm/maxWm;
