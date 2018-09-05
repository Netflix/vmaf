function calcStrredScore_opt(rname, dname, rows, colms)

band = 4;
Nscales = 5;
Nor = 6;
blk = 3;
sigma_nsq = 0.1;

srred = [];
trred = [];

rfid = fopen(rname);
dfid = fopen(dname);

iframe = 0;
while 1
    
    [yr, ~, ~] = readframefromfid(rfid, rows, colms);
    [yd, ~, ~] = readframefromfid(dfid, rows, colms);
    
    if feof(rfid) || feof(dfid)
        break;
    end
    
    % read successful, can then reshape
    yr = reshape(yr, [colms rows])';
    yd = reshape(yd, [colms rows])';
    
    if iframe > 0        
        [srred_now, ~, trred_now, ~] = extract_info_opt(yr, yr_prev, yd, ...
            yd_prev, band, Nscales, Nor, ...
            blk, sigma_nsq);
        srred = [srred srred_now];
        trred = [trred trred_now];
    end
    
    yr_prev = yr;
    yd_prev = yd;
    iframe = iframe + 1;

end

fclose(rfid);
fclose(dfid);

srred = [srred(1) srred];
trred = [trred(1) trred];

for frame_ind = 0 : iframe - 1
    
    disp(sprintf('srred: %d %f', frame_ind, srred(frame_ind + 1)));
    disp(sprintf('trred: %d %f', frame_ind, trred(frame_ind + 1)));

end;

end