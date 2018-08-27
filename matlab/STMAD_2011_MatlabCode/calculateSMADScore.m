function calcSMADScore(rname,dname,rows,colms)

rfid = fopen(rname);
dfid = fopen(dname);

iframe = -1;
while 1
    
    [yr,cbr,crr] = readframefromfid(rfid,rows,colms);
    [yd,cbr,crr] = readframefromfid(dfid,rows,colms);
    
    if feof(rfid) || feof(dfid)
        break;
    end
    
    % read successful, can then reshape
    OrgImg = reshape(yr,rows,colms)';
    DstImg = reshape(yd,rows,colms)';
    
    HiIndex = hi_index(OrgImg, DstImg);
    LoIndex = lo_index(OrgImg, DstImg);  
    
    disp(sprintf('HiIndex: %d %f', iframe, HiIndex));
    disp(sprintf('LoIndex: %d %f', iframe, LoIndex));

    iframe = iframe+1;

end

% repeat reading for last frame
disp(sprintf('HiIndex: %d %f', iframe, HiIndex));
disp(sprintf('LoIndex: %d %f', iframe, LoIndex));

fclose(rfid);
fclose(dfid);

end
  
