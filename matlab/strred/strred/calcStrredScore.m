function calcStrredScore(rname,dname,rows,colms)

rfid = fopen(rname);
dfid = fopen(dname);

srred = [];
trred = [];

iframe = 0;
while 1
    
    [yr, cbr, crr] = readframefromfid(rfid, rows, colms);
    [yd, cbr, crr] = readframefromfid(dfid, rows, colms);
    
    if feof(rfid) || feof(dfid)
        break;
    end
    
    % read successful, can then reshape
    yr = reshape(yr, rows, colms)';
    yd = reshape(yd, rows, colms)';
    
    if iframe > 0
        [spatial_ref temporal_ref] = extract_info(yr, yr_prev);
        [spatial_dis temporal_dis] = extract_info(yd, yd_prev);
        srred_now = mean2(abs(spatial_ref - spatial_dis));
        trred_now = mean2(abs(temporal_ref - temporal_dis));
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