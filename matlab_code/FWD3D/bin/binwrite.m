%binwrite(file name,variable,precision (optional, default is double))
function binwrite(varargin)
fname=varargin{1};
var=varargin{2};
S=whos('var');
prec=S.class;
 tic
 if isreal(var)
    fid=fopen([fname '.bin'],'w','l');
    fwrite(fid,var,prec);
    fclose(fid);
    compl=0;
 else
    fid=fopen([fname '.bin'],'w','l');
    fwrite(fid,real(var),prec);
    fclose(fid);
    fid=fopen([fname 'c.bin'],'w','l');
    fwrite(fid,imag(var),prec);
    fclose(fid);
    compl=1;
 end
 m=size(var,1);
 n=size(var,2);
 save([fname 'sz.mat'],'m','n','prec','compl');
 