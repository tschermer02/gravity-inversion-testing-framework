function grd=grad3d(Nx,Ny,Nz,direc)
switch direc
    case 1%dm/dx
        subD=-.5*ones(Nx,1);
        supD=-subD;
        subD(end-1)=-1;
        supD(2)=1;
        mainD=zeros(Nx,1);
        mainD(1)=-1;
        mainD(end)=1;
        gBlock=spdiags([subD,mainD,supD],[-1 0 1],Nx,Nx);
        grd=sparse(Nx*Ny*Nz,Nx*Ny*Nz);
        for iyz=1:Ny*Nz
            ind=(iyz-1)*Nx+1:iyz*Nx;
            grd(ind,ind)=grd(ind,ind)+gBlock;
        end
    case 2%dm/dy
        subD=-.5*ones(Nx*Ny,1);
        supD=-subD;
        subD(end-2*Nx+1:end)=-1;
        supD(Nx+1:2*Nx)=1;
        mainD=zeros(Nx*Ny,1);
        mainD(1:Nx)=-1;
        mainD(end-Nx+1:end)=1;
        gBlock=spdiags([subD,mainD,supD],[-Nx 0 Nx],Nx*Ny,Nx*Ny);
        grd=sparse(Nx*Ny*Nz,Nx*Ny*Nz);
        for iz=1:Nz
            ind=(iz-1)*Nx*Ny+1:iz*Nx*Ny;
            grd(ind,ind)=grd(ind,ind)+gBlock;
        end
    case 3%dm/dz
        subD=-.5*ones(Nx*Ny*Nz,1);
        supD=-subD;
        subD(end-2*Nx*Ny+1:end)=-1;
        supD(Nx*Ny+1:2*Nx*Ny)=1;
        mainD=zeros(Nx*Ny*Nz,1);
        mainD(1:Nx*Ny)=-1;
        mainD(end-Nx*Ny+1:end)=1;
        grd=spdiags([subD,mainD,supD],[-Nx*Ny 0 Nx*Ny],Nx*Ny*Nz,Nx*Ny*Nz);
end
