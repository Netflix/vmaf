function [rreds rredt]=calcStrredScore(rname,dname,nFramesPerSeg,rows,colms)

rreds=zeros(1,numel(1:nFramesPerSeg));
rredt=zeros(1,numel(1:nFramesPerSeg));
rredssn=zeros(1,numel(1:nFramesPerSeg));
rredtsn=zeros(1,numel(1:nFramesPerSeg));

for k = 1 : (nFramesPerSeg - 1)

    disp(sprintf('Process frame %d...', k));

    %Read successive frames in the reference
    [yr1,cbr,crr]=readframe(rname,k,rows,colms);
    yr1=reshape(yr1,colms,rows)';
    [yr2,cbr,crr]=readframe(rname,k+1,rows,colms);
    yr2=reshape(yr2,colms,rows)';
    %Extract temporal and spatial information from the frames
    [spatial_ref temporal_ref] = extract_info(yr1,yr2);
    
    %Read successive frames in the distorted
    [yd1,cbr,crr]=readframe(dname,k,rows,colms);
    yd1=reshape(yd1,colms,rows)';
    [yd2,cbr,crr]=readframe(dname,k+1,rows,colms);
    yd2=reshape(yd2,colms,rows)';
    %Extract temporal and spatial information from the frames
    [spatial_dis temporal_dis] = extract_info(yd1,yd2);
    
    %figure;subplot(2,2,1);imagesc(yr1);colormap gray;colorbar;subplot(2,2,2);imagesc(yd1-yr1);colormap gray;colorbar;subplot(2,2,3);imagesc(yr2);colormap gray;colorbar;subplot(2,2,4);imagesc(yd2-yr2);colormap gray;colorbar;title(sprintf('STRRED %f',strred))
    
    rreds(k) = mean2(abs(spatial_ref-spatial_dis)); %spatial RRED for frame k
    rredt(k) = mean2(abs(temporal_ref-temporal_dis));% temporal RRED for frame k
    rredssn(k) = abs(mean2(spatial_ref-spatial_dis));% spaial RRED using 1 number for frame k
    rredtsn(k) = abs(mean2(temporal_ref-temporal_dis)); % temporal RRED using 1 number for frame k

end

% pad the last frame's numbers
rreds(nFramesPerSeg) = rreds(nFramesPerSeg-1);
rredt(nFramesPerSeg) = rredt(nFramesPerSeg-1);
rredssn(nFramesPerSeg) = rredssn(nFramesPerSeg-1);
rredtsn(nFramesPerSeg) = rredtsn(nFramesPerSeg-1);

srred=mean(rreds); %spatial RRED
trred=mean(rredt); %temporal RRED
srredsn=mean(rredssn); %single no. spatial RRED
trredsn=mean(rredtsn); %single no. temporal RRED

strred = srred*trred; %spatiotemporal RRED
strredsn = srredsn*trredsn; %spatiotemporal RRED
