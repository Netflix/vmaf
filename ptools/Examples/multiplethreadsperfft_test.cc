
#include "fftw3.h"
#include <iostream>
using namespace std;

/*
 
# fftw compiled with --enable-float --enable-sse --enable-threads
# compiled on soc-dualquad1 with -O4 and FFTW_EXHAUSTIVE
# completely unloaded machine with no locking threads to processes
# (no hyperthreading on machine, 2 dual core chips, 8 processors total)
# complex to complex 1D FFT

#   threads FFTsize iterations  resulttime (wisdom computed already)

try1 1      65536   102400 
                              66.214u 0.009s 1:06.24 99.9%    0+0k 0+0io 0pf+0w
                              66.237u 0.005s 1:06.27 99.9%    0+0k 0+0io 0pf+0w
                              66.307u 0.007s 1:06.34 99.9%    0+0k 0+0io 0pf+0w
                              66.296u 0.004s 1:06.31 99.9%    0+0k 0+0io 0pf+0w

try1 2      65536   102400   
                              225.663u 3.214s 2:31.43 151.1%  0+0k 0+0io 0pf+0w
                              224.806u 3.299s 2:31.12 150.9%  0+0k 0+0io 0pf+0w
                              225.135u 3.376s 2:31.42 150.9%  0+0k 0+0io 0pf+0w
                              225.277u 3.250s 2:31.03 151.3%  0+0k 0+0io 0pf+0w

try1 4      65536   102400      
                              311.737u 6.637s 2:30.96 210.8%  0+0k 0+0io 0pf+0w
                              394.306u 8.163s 2:46.85 241.2%  0+0k 0+0io 0pf+0w
                              435.924u 8.613s 2:57.94 249.8%  0+0k 0+0io 0pf+0w
                              346.760u 7.002s 2:37.58 224.4%  0+0k 0+0io 0pf+0w


try1 1     131072   102400 
                              150.722u 0.002s 2:30.78 99.9%   0+0k 0+0io 0pf+0w
                              154.267u 0.001s 2:34.33 99.9%   0+0k 0+0io 0pf+0w
                              151.740u 0.005s 2:31.78 99.9%   0+0k 0+0io 0pf+0w
                              151.752u 0.004s 2:31.83 99.9%   0+0k 0+0io 0pf+0w

try1 2     131072   102400 
                              463.526u 3.024s 5:09.74 150.6%  0+0k 0+0io 0pf+0w
                              455.798u 2.892s 5:06.57 149.6%  0+0k 0+0io 0pf+0w
                              462.004u 2.848s 5:09.87 150.0%  0+0k 0+0io 0pf+0w

try1 1     262144   10240
                              51.360u 0.018s 0:51.39 99.9%    0+0k 0+0io 0pf+0w
                              51.593u 0.017s 0:51.62 99.9%    0+0k 0+0io 0pf+0w
                              51.207u 0.010s 0:51.23 99.9%    0+0k 0+0io 0pf+0w
                              51.453u 0.013s 0:51.48 99.9%    0+0k 0+0io 0pf+0w

try1 2     262144   10240
                              95.200u 0.460s 0:57.66 165.9%   0+0k 0+0io 0pf+0w
                              95.469u 0.406s 0:57.77 165.9%   0+0k 0+0io 0pf+0w
                              95.068u 0.476s 0:57.70 165.5%   0+0k 0+0io 0pf+0w
                              95.717u 0.387s 0:58.02 165.6%   0+0k 0+0io 0pf+0w

try1 4     262144   10240
                              156.904u 0.894s 0:57.88 272.6%  0+0k 0+0io 0pf+0w
                              152.187u 0.791s 0:56.52 270.6%  0+0k 0+0io 0pf+0w
                              158.630u 0.852s 0:58.07 274.6%  0+0k 0+0io 0pf+0w
                              136.146u 0.712s 0:53.04 258.0%  0+0k 0+0io 0pf+0w

try1  1    524288   10240 
                              141.774u 0.047s 2:21.86 99.9%   0+0k 0+0io 0pf+0w
                              141.256u 0.071s 2:21.38 99.9%   0+0k 0+0io 0pf+0w
                              140.959u 0.045s 2:21.12 99.9%   0+0k 0+0io 0pf+0w
                              141.353u 0.050s 2:21.45 99.9%   0+0k 0+0io 0pf+0w

try1  2    524288   10240 
                              210.987u 0.378s 1:58.64 178.1%  0+0k 0+0io 0pf+0w
                              211.010u 0.433s 1:58.68 178.1%  0+0k 0+0io 0pf+0w
                              210.853u 0.424s 1:58.60 178.1%  0+0k 0+0io 0pf+0w
                              210.962u 0.268s 1:58.54 178.1%  0+0k 0+0io 0pf+0w

try1  4    524288   10240 
	                      337.233u 1.243s 1:50.23 307.0%  0+0k 0+0io 0pf+0w
                              341.452u 1.299s 1:52.23 305.3%  0+0k 0+0io 0pf+0w
                              344.294u 1.332s 1:52.76 306.5%  0+0k 0+0io 0pf+0w
                              341.138u 1.301s 1:51.64 306.7%  0+0k 0+0io 0pf+0w

try1  1    786432   10240    
                              320.055u 0.020s 5:20.14 99.9%   0+0k 0+0io 0pf+0w
                              352.040u 0.218s 5:52.35 99.9%   0+0k 0+0io 0pf+0w
                              343.212u 0.062s 5:43.35 99.9%   0+0k 0+0io 0pf+0w
                              337.623u 0.157s 5:37.87 99.9%   0+0k 0+0io 0pf+0w

try1  2    786432   10240    
	                      313.925u 0.327s 2:57.32 177.2%  0+0k 0+0io 0pf+0w
                              313.083u 0.385s 2:56.91 177.1%  0+0k 0+0io 0pf+0w
                              314.222u 0.348s 2:57.53 177.1%  0+0k 0+0io 0pf+0w
                              313.721u 0.393s 2:57.30 177.1%  0+0k 0+0io 0pf+0w
	                      	
try1  4    786432   10240    

                              497.322u 1.323s 2:39.48 312.6%  0+0k 0+0io 21pf+0w                              497.270u 1.392s 2:39.51 312.6%  0+0k 0+0io 0pf+0w
                              496.335u 1.400s 2:39.64 311.7%  0+0k 0+0io 0pf+0w
	                      494.444u 1.361s 2:38.91 312.0%  0+0k 0+0io 0pf+0w


try1  1   1048576   10240    
                              344.559u 0.094s 5:44.72 99.9%   0+0k 0+0io 0pf+0w
                              342.271u 0.105s 5:42.46 99.9%   0+0k 0+0io 0pf+0w
                              341.803u 0.070s 5:41.93 99.9%   0+0k 0+0io 0pf+0w
                              342.061u 0.106s 5:42.24 99.9%   0+0k 0+0io 0pf+0w

try1  2   1048576  10240  
                              434.231u 0.360s 4:02.17 179.4%  0+0k 0+0io 0pf+0w
                              431.264u 0.497s 3:59.25 180.4%  0+0k 0+0io 0pf+0w
                              435.171u 0.498s 4:03.11 179.2%  0+0k 0+0io 0pf+0w
                              434.388u 0.479s 4:02.89 179.0%  0+0k 0+0io 0pf+0w

try1  4   1048576  10240 
                              654.358u 1.537s 3:29.24 313.4%  0+0k 0+0io 0pf+0w
                              658.603u 1.621s 3:30.67 313.3%  0+0k 0+0io 0pf+0w
                              659.853u 1.683s 3:30.72 313.9%  0+0k 0+0io 0pf+0w
                              655.172u 1.649s 3:29.82 313.0%  0+0k 0+0io 0pf+0w


try1 1   2087152  10240 
                              760.594u 0.087s 12:40.82 99.9%  0+0k 0+0io 0pf+0w
                              755.390u 0.074s 12:35.60 99.9%  0+0k 0+0io 0pf+0w
                              760.698u 0.176s 12:41.04 99.9%  0+0k 0+0io 0pf+0w
                              756.571u 0.152s 12:36.88 99.9%  0+0k 0+0io 0pf+0w

try1 2   2087152  10240 
                              942.465u 0.572s 8:39.09 181.6%  0+0k 0+0io 0pf+0w
                              936.680u 0.667s 8:35.45 181.8%  0+0k 0+0io 0pf+0w
                              940.635u 0.511s 8:39.23 181.2%  0+0k 0+0io 0pf+0w
                              944.898u 0.726s 8:39.54 182.0%  0+0k 0+0io 0pf+0w

try1 4  2097152   10240 
                              1391.302u 1.909s 7:10.19 323.8% 0+0k 0+0io 0pf+0w
                              1396.873u 2.118s 7:09.98 325.3% 0+0k 0+0io 0pf+0w
                              1400.912u 1.870s 7:10.78 325.6% 0+0k 0+0io 0pf+0w
                              1405.768u 2.004s 7:11.82 326.0% 0+0k 0+0io 0pf+0w

try1 5  2097152   10240       
                              1529.949u 2.614s 7:05.70 360.0% 0+0k 0+0io 0pf+0w
                              1532.143u 2.596s 7:05.79 360.4% 0+0k 0+0io 0pf+0w
                              1525.145u 2.631s 7:05.12 359.3% 0+0k 0+0io 0pf+0w
                              1531.745u 2.643s 7:05.74 360.4% 0+0k 0+0io 0pf+0w

try1 6 2097152 10240
                              1835.206u 3.799s 7:06.16 431.5% 0+0k 0+0io 0pf+0w
                              1833.578u 3.819s 7:06.11 431.1% 0+0k 0+0io 0pf+0w
                              1832.577u 3.737s 7:06.02 431.0% 0+0k 0+0io 0pf+0w
                              1835.108u 3.618s 7:05.97 431.6% 0+0k 0+0io 0pf+0w


try1 7 2097152 10240

                              2057.156u 5.047s 6:55.60 496.1% 0+0k 0+0io 0pf+0w
                              2055.539u 4.911s 6:55.52 495.8% 0+0k 0+0io 0pf+0w
                              2059.082u 5.143s 6:56.24 495.9% 0+0k 0+0io 0pf+0w
                              2053.995u 4.995s 6:55.50 495.5% 0+0k 0+0io 0pf+0w

try1 8 2097152 10240          
                              2481.364u 6.483s 6:43.73 616.2% 0+0k 0+0io 0pf+0w
                              2482.208u 6.712s 6:43.76 616.4% 0+0k 0+0io 0pf+0w
                              2495.153u 6.684s 6:45.49 616.9% 0+0k 0+0io 0pf+0w
                              2485.765u 6.658s 6:44.38 616.3% 0+0k 0+0io 0pf+0w
*/


void loadWisdom ()
{
  FILE* fp = fopen("./WISDOM", "r");
  if (fp==0) return;
  cout << "Loading Wisdom ..." << endl;
  fftwf_import_wisdom_from_file(fp);
  fclose(fp);
  cout << " ... done loading WISDOM" << endl;
}

void saveWisdom ()
{
  FILE* fp = fopen("./WISDOM", "w");
  fftwf_export_wisdom_to_file(fp);
  fclose(fp);
}

int main (int argc, char**argv)
{
  if (argc!=4) {
    cerr << "usage:" << argv[0] << " threads fft_size iterations" << endl;
  }
  int threads  = atoi(argv[1]);
  int N = atoi(argv[2]);
  int it = atoi(argv[3]);

  // Required to initialize threadpool for FFTW
  fftwf_init_threads();

  loadWisdom();

  // Before creating any plan, this will cause that plan
  // to use the given number of threads
  fftwf_plan_with_nthreads(threads);

 
  // Create the plans and do the computation
  {
    fftwf_complex *in, *out;
    fftwf_plan p;
    
    in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
    out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

    
    cout << "Started plan ..." << endl;
    p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_EXHAUSTIVE);
    cout << " ... plan done" << endl;
    saveWisdom();


    for (int ii=0; ii<it; ii++) {
      for (int jj=0; jj<N; jj++) {
	in[jj][0] = N;
	in[jj][1] = 0;
      }

      fftwf_execute(p);
    }

    fftwf_destroy_plan(p);

    fftwf_free(in);
    fftwf_free(out);
  }

 



  // All done
  fftwf_cleanup_threads();
}
