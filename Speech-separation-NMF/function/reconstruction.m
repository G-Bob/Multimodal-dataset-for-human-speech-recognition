function [Sc,Sr] = reconstruction(YQZ,Wc,Wr,Hc,Hr)

Xc = Wc * Hc;
Xr = Wr * Hr;
Mi = Xc < Xr;
Sr = Mi .* YQZ;
Sc = YQZ - Sr;

end

