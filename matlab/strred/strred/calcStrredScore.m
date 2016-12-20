function calcStrredScore(rname,dname,rows,colms)

rfid=fopen(rname);
dfid=fopen(dname);

iframe = -1;
while 1
    
    [yr,cbr,crr]=readframefromfid(rfid,rows,colms);
    [yd,cbr,crr]=readframefromfid(dfid,rows,colms);
    
    if feof(rfid) || feof(dfid)
        break;
    end
    
    % read successful, can then reshape
    yr=reshape(yr,colms,rows)';
    yd=reshape(yd,colms,rows)';
    
    if iframe == -1        
        yr_prev=yr;
        yd_prev=yd;
        iframe=iframe+1;
        continue;
    end
    
    [spatial_ref temporal_ref] = extract_info(yr_prev,yr);
    [spatial_dis temporal_dis] = extract_info(yd_prev,yd);

    %figure;subplot(2,2,1);imagesc(yr_prev);colormap gray;colorbar;subplot(2,2,2);imagesc(yd_prev-yr_prev);colormap gray;colorbar;subplot(2,2,3);imagesc(yr);colormap gray;colorbar;subplot(2,2,4);imagesc(yd-yr);colormap gray;colorbar;title(sprintf('STRRED %f',strred));

    srred = mean2(abs(spatial_ref-spatial_dis));
    trred = mean2(abs(temporal_ref-temporal_dis));
    
    disp(sprintf('srred: %d %f', iframe, srred));
    disp(sprintf('trred: %d %f', iframe, trred));
    
    yr_prev=yr;
    yd_prev=yd;
    iframe=iframe+1;

end

% repeat reading for last frame
disp(sprintf('srred: %d %f', iframe, srred));
disp(sprintf('trred: %d %f', iframe, trred));

fclose(rfid);
fclose(dfid);

end