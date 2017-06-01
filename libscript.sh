TOP=$(pwd)
OBJDIR=$TOP/wrapper/obj

ar rvs libvmaf.a $OBJDIR/alloc.o \
 $OBJDIR/file_io.o \
 $OBJDIR/cpu.o \
 $OBJDIR/convolution.o \
 $OBJDIR/convolution_avx.o \
 $OBJDIR/adm.o \
 $OBJDIR/adm_tools.o \
 $OBJDIR/ansnr.o \
 $OBJDIR/ansnr_tools.o \
 $OBJDIR/vif.o \
 $OBJDIR/vif_tools.o \
 $OBJDIR/motion.o \
 $OBJDIR/psnr.o \
 $OBJDIR/math_utils.o \
 $OBJDIR/convolve.o \
 $OBJDIR/decimate.o \
 $OBJDIR/ssim_tools.o \
 $OBJDIR/ssim.o \
 $OBJDIR/ms_ssim.o \
 $OBJDIR/svm.o \
 $OBJDIR/combo.o \
 $OBJDIR/vmaf.o \
 $OBJDIR/darray.o \
 $OBJDIR/main.o \
 $OBJDIR/pugixml.o \
