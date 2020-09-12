function run_icid(ref_filename, dis_filename, width, height, yuv_type)

path(path,'./iCID/iCID_Metric');
path(path,'./iCID/iCID_Metric/ColorSpaceTransformations');

yuv_ref = yuvread(ref_filename,width,height,yuv_type,1);
yuv_dis = yuvread(dis_filename,width,height,yuv_type,1);

len = min(length(yuv_ref), length(yuv_dis));

result = zeros(1,len);

for idx = 1:len
    OrgImg = yuv_ref(idx).cdata;
    DstImg = yuv_dis(idx).cdata;
    OrgImg = yuv2rgb(OrgImg(:,:,1), OrgImg(:,:,2), OrgImg(:,:,3),'YUV444_8');
    DstImg = yuv2rgb(DstImg(:,:,1), DstImg(:,:,2), DstImg(:,:,3),'YUV444_8');
    [result(idx), ~] = iCID(OrgImg, DstImg);
end


for frame_ind = 0 : len - 1
    disp(sprintf('icid: %d %f', frame_ind, result(frame_ind + 1)));
end;

end
