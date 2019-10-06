
#include "pipelinefft.h"

// Timings for higher workloads
// -O4 Queue size of 5, FFT_ESTIMATE soc-dualwuad1, uses local memories of 16M

// time ./pipelinefft_test 200 4194304 4194304 1
//    92.618u 9.514s 1:25.24 119.8%   0+0k 0+0io 29pf+0w
// time ./pipelinefft_test 200 4194304 4194304 2
//    123.611u 10.065s 0:58.16 229.8% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 3
//    139.763u 12.907s 0:43.23 353.1% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 4
//   165.435u 16.531s 0:40.71 446.9% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 5
//   205.287u 17.754s 0:39.99 557.7% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 6
//   241.374u 18.651s 0:40.39 643.7% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 7
//   277.078u 20.505s 0:40.29 738.5% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 8
//   280.169u 20.762s 0:41.27 729.1% 0+0k 0+0io 0pf+0w

// -O4 Queue size of 8, FFT_ESTIMATE soc-dualwuad1, uses local memories of 1Gig

// time ./pipelinefft_test 200 4194304 4194304 1
//    92.618u 9.514s 1:25.24 119.8%   0+0k 0+0io 29pf+0w
// time ./pipelinefft_test 200 4194304 4194304 2
//   112.687u 11.396s 0:53.60 231.4% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 3
//   148.137u 13.007s 0:46.99 342.9% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 4
//   176.617u 16.919s 0:45.46 425.6% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 5
//   217.128u 17.805s 0:43.17 544.1% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 6
//   252.307u 19.810s 0:42.42 641.4% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 7
//   297.314u 22.226s 0:42.46 752.5% 0+0k 0+0io 0pf+0w
// time ./pipelinefft_test 200 4194304 4194304 8
//   295.664u 21.175s 0:43.17 733.9% 0+0k 0+0io 0pf+0w


/// Use the timings HERE, not at the end of the file.
// These are for -O4, Queue of 10, FFT_EXHAUSTIVE, soc-dualquad1

// ***** time baseline_test 20000 16384
//   4.261u 0.129s 0:02.62 167.1%    0+0k 0+0io 0pf+0w
//   2.791u 0.104s 0:01.83 157.9%    0+0k 0+0io 0pf+0w
//   4.252u 0.118s 0:02.64 165.1%    0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 1
//    13.462u 0.602s 0:07.53 186.7%   0+0k 0+0io 0pf+0w
//    13.061u 0.736s 0:08.25 167.1%   0+0k 0+0io 0pf+0w
//    11.208u 0.620s 0:07.43 159.0%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 2
//    14.746u 3.285s 0:05.52 326.4%   0+0k 0+0io 0pf+0w
//    14.621u 2.970s 0:05.61 313.5%   0+0k 0+0io 0pf+0w
//    15.137u 3.145s 0:05.63 324.5%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 3
//    13.388u 1.001s 0:04.34 331.3%   0+0k 0+0io 0pf+0w
//    13.494u 1.092s 0:03.98 366.3%   0+0k 0+0io 0pf+0w
//    14.847u 1.539s 0:04.32 378.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 4
//    15.710u 2.684s 0:04.36 421.7%   0+0k 0+0io 0pf+0w
//    15.715u 2.775s 0:04.30 429.7%   0+0k 0+0io 0pf+0w
//    15.587u 2.836s 0:04.27 431.1%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 5
//    16.150u 3.192s 0:04.39 440.5%   0+0k 0+0io 0pf+0w
//    16.921u 3.162s 0:04.63 433.6%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 6
//    18.282u 3.214s 0:04.40 488.4%   0+0k 0+0io 0pf+0w
//    18.739u 3.081s 0:04.38 497.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 7
//    18.081u 2.975s 0:04.57 460.6%   0+0k 0+0io 0pf+0w
//    16.818u 2.394s 0:05.03 381.7%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 8
//    15.705u 2.251s 0:04.91 365.5%   0+0k 0+0io 0pf+0w
//    18.203u 3.035s 0:04.97 427.1%   0+0k 0+0io 0pf+0w


// ***** time baseline_test 20000 32768
//  8.072u 0.162s 0:04.85 169.6%    0+0k 0+0io 0pf+0w
//  7.875u 0.177s 0:04.83 166.4%    0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 1
//   22.582u 0.682s 0:12.74 182.5%   0+0k 0+0io 0pf+0w
//   25.529u 0.707s 0:12.55 208.9%   0+0k 0+0io 0pf+0w
//   23.118u 0.867s 0:13.38 179.1%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 2
//  27.819u 4.810s 0:10.19 320.1%   0+0k 0+0io 0pf+0w
//  27.771u 4.745s 0:09.61 338.2%   0+0k 0+0io 0pf+0w
//  27.317u 5.178s 0:09.73 333.8%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 3
//  29.106u 3.917s 0:07.68 429.8%   0+0k 0+0io 0pf+0w
//  29.037u 3.720s 0:07.83 418.2%   0+0k 0+0io 0pf+0w
//  27.834u 2.000s 0:08.05 370.5%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 4
//   31.481u 4.142s 0:08.21 433.8%   0+0k 0+0io 0pf+0w
//   32.113u 3.950s 0:07.80 462.3%   0+0k 0+0io 0pf+0w
//   30.157u 3.935s 0:07.97 427.6%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 5
//  30.461u 4.717s 0:08.19 429.4%   0+0k 0+0io 0pf+0w
//  30.858u 4.731s 0:08.40 423.5%   0+0k 0+0io 0pf+0w
//  30.787u 4.684s 0:08.34 425.1%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 6
//  34.400u 3.971s 0:08.27 463.9%   0+0k 0+0io 0pf+0w
//  33.574u 4.206s 0:08.44 447.5%   0+0k 0+0io 0pf+0w
//  35.939u 3.886s 0:08.03 495.7%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 7
//  31.997u 4.149s 0:08.66 417.2%   0+0k 0+0io 0pf+0w
//  34.037u 4.417s 0:08.90 431.9%   0+0k 0+0io 0pf+0w
//  33.256u 3.500s 0:08.76 419.5%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 8
//  33.756u 4.384s 0:09.06 420.8%   0+0k 0+0io 0pf+0w
//  32.461u 4.173s 0:09.23 396.8%   0+0k 0+0io 0pf+0w
//  33.445u 4.047s 0:08.23 455.4%   0+0k 0+0io 0pf+0w


// ***** time baseline_test 20000 65536
//  14.073u 0.235s 0:07.90 181.0%   0+0k 0+0io 0pf+0w
//  14.167u 0.204s 0:07.92 181.3%   0+0k 0+0io 0pf+0w
//  13.819u 0.732s 0:08.10 179.5%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 1
//  47.120u 1.368s 0:25.72 188.4%   0+0k 0+0io 0pf+0w
//  47.426u 1.310s 0:26.56 183.4%   0+0k 0+0io 0pf+0w
//  50.762u 1.473s 0:27.73 188.3%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 2
//  55.292u 9.002s 0:18.86 340.8%   0+0k 0+0io 0pf+0w
//  54.822u 7.808s 0:18.33 341.6%   0+0k 0+0io 0pf+0w
//  54.621u 9.186s 0:18.63 342.4%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 3
//  61.023u 9.860s 0:16.34 433.7%   0+0k 0+0io 0pf+0w
//  61.287u 9.703s 0:16.19 438.4%   0+0k 0+0io 0pf+0w
//  60.602u 10.754s 0:16.78 425.2%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 4
//  75.216u 9.392s 0:16.50 512.7%   0+0k 0+0io 0pf+0w
//  75.632u 10.430s 0:16.93 508.3%  0+0k 0+0io 0pf+0w
//  76.556u 10.580s 0:16.92 514.9%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 5
//  76.283u 9.426s 0:15.52 552.1%   0+0k 0+0io 0pf+0w
// 68.147u 8.762s 0:15.78 487.3%   0+0k 0+0io 0pf+0w
//  66.760u 9.509s 0:16.06 474.8%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 6
//  72.817u 8.225s 0:16.44 492.8%   0+0k 0+0io 0pf+0w
//  64.947u 5.431s 0:15.84 444.2%   0+0k 0+0io 0pf+0w
//  64.628u 7.646s 0:15.84 456.1%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 7
//  67.943u 6.826s 0:17.68 422.8%   0+0k 0+0io 0pf+0w
//  67.431u 6.882s 0:17.35 428.2%   0+0k 0+0io 0pf+0w
//  71.759u 7.841s 0:16.92 470.3%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 8
//  67.165u 7.949s 0:16.81 446.7%   0+0k 0+0io 0pf+0w
//  69.313u 7.207s 0:17.31 441.9%   0+0k 0+0io 0pf+0w
//  72.893u 7.496s 0:16.66 482.4%   0+0k 0+0io 0pf+0w


// ***** time baseline_test 20000 131072
//  26.831u 0.308s 0:14.26 190.2%   0+0k 0+0io 0pf+0w
//  26.855u 0.332s 0:14.27 190.4%   0+0k 0+0io 0pf+0w
//  24.557u 0.309s 0:13.51 183.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 1
//   69.731u 1.062s 0:55.26 128.1%   0+0k 0+0io 0pf+0w
//   99.463u 1.603s 0:55.40 182.4%   0+0k 0+0io 0pf+0w
//  109.488u 1.484s 0:51.88 213.8%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 2
//   110.551u 16.353s 0:36.71 345.6% 0+0k 0+0io 0pf+0w
//   112.138u 15.112s 0:36.88 345.0% 0+0k 0+0io 0pf+0w
//   111.844u 18.328s 0:39.31 331.1% 0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 3
//  127.392u 24.867s 0:35.32 431.0% 0+0k 0+0io 0pf+0w



// time pipelinefft_test 20000 131072 16384 4

// time pipelinefft_test 20000 131072 16384 5

// time pipelinefft_test 20000 131072 16384 6

// time pipelinefft_test 20000 131072 16384 7

// time pipelinefft_test 20000 131072 16384 8

// time pipelinefft_test 20000 131072 16384 9


int main (int argc, char**argv)
{
  if (argc!=4 && argc!=5) {
    cerr << "usage:" << argv[0] << " iterations packetsize fftsize [threads]" << endl;
    exit(1);
  }
  int_8 iterations  = atoi(argv[1]);
  int_8 length  = atoi(argv[2]);
  int_8 fftsize = atoi(argv[3]);
  int_8 threads = (argc==5) ? atoi(argv[4]) : 2;

  // create
  Constant c("constant");
  c.set("iter", iterations);
  c.set("DataLength", length);

  PipelineFFT m("pipelineFFT");
  m.set("iter", iterations);
  m.set("FFTSize", fftsize);
  m.set("Threads", threads);

  //MaxMin  mm("maxmin")
  Empty  mm("maxmin");
  mm.set("iter", iterations);

  // connect
  CQ* a = new CQ(4);
  CQ* b = new CQ(4);
  c.connect(0, a);
  m.connect(a, b);
  mm.connect(b, 0);

  // start
  mm.start();
  m.start();
  c.start();

  // Wait for everyone to finish
  mm.wait(); 
  m.wait();  
  c.wait();  

  cout << "Max = " << mm.get("Max") << endl;
  cout << "Min = " << mm.get("Min") << endl;

  delete a;
  delete b;
}

/// OOOOOLLLLLDD TTESTS WITH A QUEUE SIZE OF 4 SO LIMITED PARALLELISM!!!!
///  USE THE TIMINGS AT THE TOP OF THE FILE!!!!


// time pipelinefft_test 20000 16384 16384 1
//  13.288u 0.414s 0:08.73 156.8%   0+0k 0+0io 28pf+0w
//  13.487u 0.449s 0:08.55 162.8%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 2
//  15.355u 1.423s 0:05.11 328.1%   0+0k 0+0io 0pf+0w
//  14.734u 2.263s 0:05.51 308.3%   0+0k 0+0io 0pf+0w
//  15.265u 2.821s 0:05.69 317.7%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 3
//   14.845u 2.214s 0:03.76 453.4%   0+0k 0+0io 0pf+0w
//   15.465u 0.594s 0:03.91 410.4%   0+0k 0+0io 0pf+0w
//   14.168u 0.539s 0:03.74 392.7%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 4
//   13.808u 2.775s 0:03.99 415.2%   0+0k 0+0io 0pf+0w
//   13.709u 2.712s 0:03.78 434.1%   0+0k 0+0io 0pf+0w
//   13.940u 3.031s 0:04.17 406.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 5
//   13.933u 2.430s 0:04.31 379.5%   0+0k 0+0io 0pf+0w
//   13.646u 2.735s 0:04.81 340.3%   0+0k 0+0io 0pf+0w
//   13.735u 2.610s 0:04.54 359.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 6 
//   14.758u 2.672s 0:03.87 450.1%   0+0k 0+0io 0pf+0w
//   14.852u 2.389s 0:04.21 409.2%   0+0k 0+0io 0pf+0w
//   15.813u 2.967s 0:04.02 466.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 7 
//   15.828u 2.377s 0:03.91 465.2%   0+0k 0+0io 0pf+0w
//   14.115u 1.613s 0:03.84 409.3%   0+0k 0+0io 0pf+0w
// 14.452u 1.198s 0:03.53 443.0%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 16384 16384 8
// 14.176u 2.118s 0:04.35 374.2%   0+0k 0+0io 0pf+0w
// 14.482u 0.916s 0:03.49 440.9%   0+0k 0+0io 0pf+0w
// 15.196u 1.270s 0:04.15 396.6%   0+0k 0+0io 0pf+0w


// time pipelinefft_test 20000 32768 16384 1
//  23.239u 0.679s 0:16.19 147.6%   0+0k 0+0io 0pf+0w
//  27.267u 0.551s 0:17.31 160.6%   0+0k 0+0io 0pf+0w
//  22.902u 1.311s 0:14.79 163.6%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 2
//   28.405u 4.701s 0:10.69 309.6%   0+0k 0+0io 0pf+0w
//   28.188u 4.461s 0:10.15 321.5%   0+0k 0+0io 0pf+0w
//   30.009u 2.403s 0:10.16 318.8%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 3
//   26.845u 1.665s 0:07.19 396.3%   0+0k 0+0io 0pf+0w
//   27.225u 1.672s 0:07.01 412.1%   0+0k 0+0io 0pf+0w
//   26.701u 1.641s 0:06.84 414.3%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 4
//   26.568u 4.792s 0:07.26 431.8%   0+0k 0+0io 0pf+0w
//   27.013u 5.695s 0:07.56 432.5%   0+0k 0+0io 0pf+0w
//   27.540u 5.080s 0:07.95 410.3%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 5
//   27.004u 5.145s 0:07.51 427.9%   0+0k 0+0io 0pf+0w
//   28.407u 2.086s 0:06.94 439.1%   0+0k 0+0io 0pf+0w
//   27.329u 5.435s 0:07.41 441.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 6
//   28.972u 1.969s 0:06.96 444.3%   0+0k 0+0io 0pf+0w
//   31.580u 5.084s 0:07.48 490.1%   0+0k 0+0io 0pf+0w
//   30.372u 2.565s 0:06.65 495.1%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 7
//   30.161u 4.370s 0:08.66 398.7%   0+0k 0+0io 0pf+0w
//   30.171u 4.088s 0:07.33 467.2%   0+0k 0+0io 0pf+0w
//   30.914u 2.085s 0:07.08 465.9%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 32768 16384 8
//   29.690u 4.029s 0:09.01 374.1%   0+0k 0+0io 0pf+0w
//   29.505u 2.147s 0:06.84 462.5%   0+0k 0+0io 0pf+0w
//   29.211u 2.181s 0:06.84 458.9%   0+0k 0+0io 0pf+0w


// time pipelinefft_test 20000 65536 16384 1
//   37.861u 0.387s 0:28.19 135.6%   0+0k 0+0io 0pf+0w
//   43.870u 1.561s 0:28.20 161.0%   0+0k 0+0io 0pf+0w
//   51.854u 1.854s 0:33.45 160.5%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 2
//   57.615u 7.999s 0:20.84 314.7%   0+0k 0+0io 0pf+0w
//   57.861u 6.391s 0:20.76 309.4%   0+0k 0+0io 0pf+0w
//   60.386u 4.055s 0:20.42 315.5%   0+0k 0+0io 0pf+0w


// time pipelinefft_test 20000 65536 16384 3
//   54.498u 3.203s 0:13.54 426.0%   0+0k 0+0io 0pf+0w 
//   54.408u 2.892s 0:13.37 428.4%   0+0k 0+0io 0pf+0w
//   54.061u 2.127s 0:13.56 414.3%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 4
//   52.476u 11.634s 0:14.35 446.6%  0+0k 0+0io 0pf+0w
//   55.456u 10.351s 0:14.56 451.9%  0+0k 0+0io 0pf+0w
//   54.860u 10.563s 0:14.39 454.6%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 5
//   

// time pipelinefft_test 20000 65536 16384 6
// time pipelinefft_test 20000 65536 16384 7
// time pipelinefft_test 20000 65536 16384 8



//// THESE ARE ALL DONE WITH ESTIMATE
// time pipelinefft_test 20000 65536 16384 1
//   59.739u 1.067s 0:32.00 189.9%   0+0k 0+0io 0pf+0w
//   59.904u 0.545s 0:31.63 191.0%   0+0k 0+0io 0pf+0w
//   58.646u 0.933s 0:31.55 188.8%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 2
//   58.551u 3.525s 0:18.40 337.3%   0+0k 0+0io 0pf+0w
//   60.720u 7.266s 0:20.97 324.1%   0+0k 0+0io 0pf+0w
//   59.143u 5.190s 0:19.68 326.8%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 3
//   62.419u 6.145s 0:14.93 459.1%   0+0k 0+0io 0pf+0w
//   63.182u 4.928s 0:15.10 450.9%   0+0k 0+0io 0pf+0w
//   62.480u 7.252s 0:15.15 460.2%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 4
//  63.578u 5.007s 0:14.05 488.0%   0+0k 0+0io 0pf+0w
//  61.945u 3.318s 0:14.05 464.4%   0+0k 0+0io 0pf+0w
//  62.597u 3.697s 0:14.02 472.7%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 5
//  63.379u 5.985s 0:13.74 504.7%   0+0k 0+0io 0pf+0w
//  60.647u 5.287s 0:14.08 468.1%   0+0k 0+0io 0pf+0w
//  63.715u 6.107s 0:13.85 504.0%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 65536 16384 6
//  68.934u 4.502s 0:14.78 496.8%   0+0k 0+0io 0pf+0w
//  67.413u 3.865s 0:14.53 490.5%   0+0k 0+0io 0pf+0w
//  71.457u 4.265s 0:14.21 532.7%   0+0k 0+0io 0pf+0w


// time pipelinefft_test 20000 131072 16384 1
//   102.210u 2.483s 1:06.33 157.8%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 2
//   115.723u 3.182s 0:33.44 355.5%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 3
//   133.279u 13.670s 0:32.18 456.6% 0+0k 0+0io 0pf+0w
//   128.029u 19.099s 0:33.70 436.5% 0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 4
//    159.492u 14.413s 0:32.78 530.5% 0+0k 0+0io 0pf+0w
//    150.500u 10.643s 0:31.16 517.1% 0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 131072 16384 8
//    147.268u 9.112s 0:30.90 506.0%  0+0k 0+0io 0pf+0w
//    151.869u 9.087s 0:31.38 512.8%  0+0k 0+0io 0pf+0w


//******* baseline_test 20000 262144 
//    51.889u 2.939s 0:28.56 191.9%   0+0k 0+0io 0pf+0w
//    52.829u 0.586s 0:27.20 196.3%   0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 262144 16384 1
//    248.159u 1.493s 2:12.73 188.0%  0+0k 0+0io 0pf+0w
//    224.226u 4.439s 2:17.17 166.6%  0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 262144 16384 2
//    271.367u 4.605s 1:24.33 327.2%  0+0k 0+0io 0pf+0w
//    263.187u 10.774s 1:24.35 324.7% 0+0k 0+0io 0pf+0w

// time pipelinefft_test 20000 262144 16384 3
//    269.250u 11.949s 1:04.09 438.7% 0+0k 0+0io 0pf+0w
//    


