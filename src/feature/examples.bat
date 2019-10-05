rem refer feature/examples

vmaf
echo ""

echo "run adm:"
vmaf adm yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run ansnr:"
vmaf ansnr yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run motion:"
vmaf motion yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run vif:"
vmaf vif yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run all:"
vmaf all yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run psnr:"
psnr yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run 2nd moment:"
moment 2 yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv 576 324

echo "run ssim:"
ssim yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "run ms_ssim:"
ms_ssim yuv420p ..\..\python\test\resource\yuv\src01_hrc00_576x324.yuv ..\..\python\test\resource\yuv\src01_hrc01_576x324.yuv 576 324

echo "done."
