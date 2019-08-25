function calcSMADScore(rname,dname,colms,rows)

rfid = fopen(rname);
dfid = fopen(dname);

iframe = 0;
while 1
    
    [yr,~,~] = readframefromfid(rfid,rows,colms);
    [yd,~,~] = readframefromfid(dfid,rows,colms);
    
    if feof(rfid) || feof(dfid)
        break;
    end
    
    % read successful, can then reshape
    OrgImg = reshape(yr,[colms rows])';
    DstImg = reshape(yd,[colms rows])';
    
    HiIndex = hi_index(OrgImg, DstImg);
    LoIndex = lo_index(OrgImg, DstImg);  

    disp(sprintf('HiIndex: %d %f', iframe, HiIndex));
    disp(sprintf('LoIndex: %d %f', iframe, LoIndex));
    
    iframe = iframe+1;

end

% repeat reading for last frame
% disp(sprintf('HiIndex: %d %f', iframe, HiIndex));
% disp(sprintf('LoIndex: %d %f', iframe, LoIndex));

fclose(rfid);
fclose(dfid);

end
  
