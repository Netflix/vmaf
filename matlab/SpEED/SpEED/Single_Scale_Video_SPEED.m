function [speed_s, speed_s_sn, speed_t, speed_t_sn] = ...
    Single_Scale_Video_SPEED(ref, ref_next, dis, dis_next, times_to_down_size, window, blk, sigma_nsq)

%%%% resize all frames
for band_ind = 1 : times_to_down_size
    ref = imresize(ref, 0.5);
    ref_next = imresize(ref_next, 0.5);
    dis = imresize(dis, 0.5);
    dis_next = imresize(dis_next, 0.5);
end
%%%% calculate local averages
mu_ref = imfilter(ref, window, 'replicate');
mu_dis = imfilter(dis, window, 'replicate');

%%%% Spatial SpEED
%%%% estimate local variances and conditional entropies in the spatial
%%%% domain for ith reference and distorted frames
[ss_ref, q_ref] = est_params(ref - mu_ref, blk, sigma_nsq);
spatial_ref = q_ref.*log2(1+ss_ref);
[ss_dis, q_dis] = est_params(dis - mu_dis, blk, sigma_nsq);
spatial_dis = q_dis.*log2(1+ss_dis);
speed_s = nanmean(abs(spatial_ref(:) - spatial_dis(:)));
speed_s_sn = abs(nanmean(spatial_ref(:) - spatial_dis(:)));

%%%% frame differencing
ref_diff = ref_next - ref;
dis_diff = dis_next - dis;
%%%% calculate local averages of frame differences
mu_ref_diff = imfilter(ref_diff, window, 'replicate');
mu_dis_diff = imfilter(dis_diff, window, 'replicate');

%%%% Temporal SpEED
%%%% estimate local variances and conditional entropies in the spatial
%%%% domain for the reference and distorted frame differences
[ss_ref_diff, q_ref] = est_params(ref_diff - mu_ref_diff, blk, sigma_nsq);
temporal_ref = q_ref.*log2(1+ss_ref).*log2(1+ss_ref_diff);
[ss_dis_diff, q_dis] = est_params(dis_diff - mu_dis_diff, blk, sigma_nsq);
temporal_dis = q_dis.*log2(1+ss_dis).*log2(1+ss_dis_diff);
speed_t = nanmean(abs(temporal_ref(:) - temporal_dis(:)));
speed_t_sn = abs(nanmean(temporal_ref(:) - temporal_dis(:)));

end