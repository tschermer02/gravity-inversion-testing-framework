function [var]=binread(fname)

load([fname 'sz.mat']);
sz=m*n;
fid=fopen([fname '.bin'],'r','l');
%if strcmp('double',prec);
    var=fread(fid,[prec '=>' prec]);
% else
%     var=fread(fid,'single=>single');
% end
fclose(fid);

if compl 
    fid=fopen([fname 'c.bin'],'r','l');
%if strcmp('double',prec);
    varc=fread(fid,[prec '=>' prec]);
%else
%    varc=fread(fid,'single=>single');
%end
fclose(fid);
var=var+1i.*varc;
clear varc;
end
var=reshape(var,m,n);

