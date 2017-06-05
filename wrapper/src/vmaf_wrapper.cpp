#include "vmaf.h"
#include "vmaf_wrapper.h"
#include "cpu.h"

enum vmaf_cpu cpu; // global

extern "C" {
	/*Asset* newAsset() {
		return new Asset();
	}

	void Asset1(Asset* v,int w, int h, const char *ref_path, const char *dis_path, const char *fmt){
		v->w = w;
		v->h = h;
		v->ref_path = ref_path;
		v->dis_path = dis_path;
		v->fmt = fmt;
	}

	void Asset2(Asset* v,int w, int h, const char *ref_path, const char *dis_path){
		v->w = w;
		v->h = h;
		v->ref_path = ref_path;
		v->dis_path = dis_path;
		v->fmt = "yuv420p";
	}

	int get_width(Asset* v) {
	   	return v->get_width();
	}

	int get_height(Asset* v) {
		return v->get_height();
	}

	const char* get_ref_path(Asset* v){
		return v->get_ref_path();
	};

	const char* get_dis_path(Asset* v){
		return v->get_dis_path();
	};

	const char* get_fmt(Asset* v){
		return v->get_fmt();
	};

	void deleteMyClass(Asset* v) {
		delete v;
	}*/


	double RunVmaf1(const char* fmt, int width, int height,
		           const char *ref_path, const char *dis_path, const char *model_path,
		           const char *log_path, const char *log_fmt,
		           int disable_clip, int enable_transform,
		           int do_psnr, int do_ssim, int do_ms_ssim,
		           const char *pool_method){   
		
		bool d_c = false;
		bool disable_avx = false;
		bool e_t = false;
		bool d_p = false;
		bool d_s = false;
		bool d_m_s = false;

		if(disable_clip){
			d_c = true;	
		}
		if(enable_transform){
			e_t = true;	
		}
		if(do_psnr){
			d_p = true;	
		}
		if(do_ssim){
			d_s = true;	
		}
		if(do_ms_ssim){
			d_m_s = true;	
		}
		
        cpu = cpu_autodetect();

        if (disable_avx)
        {
            cpu = VMAF_CPU_NONE;
        }

		printf("under vmaf_wrapper\n");

		try
		{
		    double score = RunVmaf(fmt, width, height, ref_path, dis_path, model_path, log_path, log_fmt, d_c, e_t, d_p, d_s, d_m_s, pool_method);
			printf("after runvmaf under vmaf_wrapper\n");
		    return score;
		}
		catch (const std::exception &e)
		{
		    fprintf(stderr, "Error: %s\n", e.what());
		    return -1;
		}
	

	}


}
