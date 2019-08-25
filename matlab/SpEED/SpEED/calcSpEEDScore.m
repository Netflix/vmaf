function calcSpEEDScore(rname, dname, colms, rows, bands, yuv_type)

blk = 5;
sigma_nsq = 0.1;
window = fspecial('gaussian', 7, 7/6);
window = window/sum(sum(window));
for band = bands
    
    sspeed = [];
    tspeed = [];
    
    rfid=fopen(rname);
    dfid=fopen(dname);
    
    iframe = 0;
    while 1
        
        [yr, ~, ~] = readframefromfid_all_fmts(rfid, rows, colms, yuv_type);
        [yd, ~, ~] = readframefromfid_all_fmts(dfid, rows, colms, yuv_type);
        
        if feof(rfid) || feof(dfid)
            break;
        end
        
        % read successful, can then reshape
        yr=reshape(yr, [colms rows])';
        yd=reshape(yd, [colms rows])';
        
        if iframe > 0
            [sspeed_now, ~, tspeed_now, ~] = ...
                Single_Scale_Video_SPEED(yr, yr_prev, yd, yd_prev, band, window, blk, sigma_nsq);
            sspeed = [sspeed sspeed_now];
            tspeed = [tspeed tspeed_now];
        end
        
        yr_prev=yr;
        yd_prev=yd;
        iframe=iframe+1;
        
    end
    
    fclose(rfid);
    fclose(dfid);
    
    sspeed = [sspeed(1) sspeed];
    tspeed = [tspeed(1) tspeed];
    
    for frame_ind = 0 : iframe - 1
        
        fprintf('sspeed_%s: %d %f\n', num2str(band), frame_ind, sspeed(frame_ind+1));
        fprintf('tspeed_%s: %d %f\n', num2str(band), frame_ind, tspeed(frame_ind+1));
        
    end
    
end