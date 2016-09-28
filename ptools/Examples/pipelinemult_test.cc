
#include "pipelinemult.h"

// This program tests how fast we can move data through the system and
// just multiply each packet.

//////////////////// baseline_test 200 4194304 
/// 9.191u 5.654s 0:10.75 138.0%    0+0k 0+0io 0pf+0w

// Even with big pipes, STILL not enough work for a pipeline soln

//   -O4 soc-dualquad1
// time ./pipelinemult_test 200 4194304 10 1
//   28.839u 9.315s 0:17.20 221.7%   0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 2
//   28.939u 9.330s 0:17.22 222.1%   0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 3
//   30.641u 9.108s 0:17.16 231.5%   0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 4
//  28.392u 9.805s 0:17.36 219.9%   0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 5
//  28.115u 10.172s 0:17.49 218.8%  0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 6
//  27.949u 10.405s 0:17.40 220.3%  0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 7
//  29.723u 9.429s 0:17.31 226.1%   0+0k 0+0io 0pf+0w

// time ./pipelinemult_test 200 4194304 10 8
// 29.767u 9.480s 0:17.31 226.6%   0+0k 0+0io 0pf+0w


///  ALL THE TESTS BELOW SHOW YOU NEED BIG PACKETS, OTHERWISE
// YOU SEE LITTLE IMPROVEMENT
// Times recorded on Dec. 18th, 2008  -O4 soc-dualquad1


// Using MaxMin as the final analyzer, Queue size of 4


// time pipelinemult_test 20000 16384 10 1
//  10.469u 0.682s 0:04.91 226.8%   0+0k 0+0io 0pf+0w

// time  pipelinemult_test 20000 16384 10 2
//   9.568u 1.570s 0:03.78 294.4%    0+0k 0+0io 0pf+0w
//   8.003u 1.314s 0:03.76 247.6%    0+0k 0+0io 0pf+0w

// time  pipelinemult_test 20000 16384 10 3
//   8.668u 0.843s 0:02.85 333.3%    0+0k 0+0io 0pf+0w
//   9.832u 0.827s 0:03.56 299.1%    0+0k 0+0io 0pf+0w
//   9.276u 0.823s 0:03.40 296.7%    0+0k 0+0io 0pf+0w

// time  pipelinemult_test 20000 16384 10 4
//   8.333u 1.688s 0:03.48 287.6%    0+0k 0+0io 0pf+0w
//   7.859u 2.020s 0:03.71 266.0%    0+0k 0+0io 0pf+0w
//   9.010u 2.857s 0:04.11 288.5%    0+0k 0+0io 0pf+0w

// time  pipelinemult_test 20000 16384 10 8
//   8.650u 2.368s 0:03.97 277.3%    0+0k 0+0io 0pf+0w
//   8.413u 0.443s 0:03.49 253.5%    0+0k 0+0io 0pf+0w



// time pipelinemult_test 20000 32768 10 1
//   19.386u 0.827s 0:08.68 232.7%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 32768 10 2
//   15.172u 1.694s 0:06.62 254.6%   0+0k 0+0io 0pf+0w
//   17.811u 2.873s 0:07.10 291.2%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 32768 10 3
//  19.445u 1.424s 0:05.71 365.3%   0+0k 0+0io 0pf+0w
//  18.957u 1.875s 0:07.77 267.9%   0+0k 0+0io 0pf+0w
//  19.554u 2.062s 0:07.35 294.0%   0+0k 0+0io 0pf+0w


// time pipelinemult_test 20000 32768 10 4
//   17.350u 4.062s 0:08.29 258.2%   0+0k 0+0io 0pf+0w
//   16.153u 3.626s 0:07.19 274.9%   0+0k 0+0io 0pf+0w
//   15.193u 4.975s 0:07.36 273.9%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 32768 10 6
//   16.812u 2.845s 0:06.68 294.1%   0+0k 0+0io 0pf+0w
//   18.036u 2.954s 0:07.12 294.6%   0+0k 0+0io 0pf+0w


// time pipelinemult_test 20000 32768 10 8
//   17.501u 2.477s 0:06.98 286.1%   0+0k 0+0io 0pf+0w
//   18.699u 2.366s 0:06.90 305.0%   0+0k 0+0io 0pf+0w


// time pipelinemult_test 20000 65536 10 1
//   35.512u 1.618s 0:16.72 222.0%   0+0k 0+0io 0pf+0w
//   36.284u 2.176s 0:16.55 232.3%   0+0k 0+0io 0pf+0w
//   36.818u 1.853s 0:17.29 223.5%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 65536 10 2
//   34.418u 5.108s 0:13.41 294.6%   0+0k 0+0io 0pf+0w
//   35.777u 5.347s 0:13.55 303.3%   0+0k 0+0io 0pf+0w
//   36.618u 4.611s 0:13.46 306.2%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 65536 10 3
//   39.358u 2.013s 0:13.67 302.5%   0+0k 0+0io 0pf+0w
//   37.423u 3.012s 0:13.54 298.5%   0+0k 0+0io 0pf+0w
//   34.283u 3.154s 0:13.51 277.0%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 65536 10 4
//   33.163u 7.806s 0:13.74 298.1%   0+0k 0+0io 0pf+0w
//   31.267u 9.612s 0:13.95 292.9%   0+0k 0+0io 0pf+0w



// time pipelinemult_test 20000 131072 10 1
//   79.085u 2.372s 0:34.53 235.8%   0+0k 0+0io 0pf+0w
//   69.948u 0.702s 0:34.42 205.2%   0+0k 0+0io 0pf+0w
//   80.593u 1.824s 0:32.27 255.3%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 131072 10 2
//   82.327u 8.797s 0:27.28 333.9%   0+0k 0+0io 0pf+0w
//   77.920u 12.797s 0:28.56 317.6%  0+0k 0+0io 0pf+0w
//   80.716u 10.056s 0:27.20 333.6%  0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 131072 10 3
//   80.407u 5.869s 0:28.41 303.6%   0+0k 0+0io 0pf+0w
//   80.695u 6.167s 0:28.38 306.0%   0+0k 0+0io 0pf+0w
//   83.836u 4.557s 0:28.23 313.0%   0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 131072 10 4
//   71.826u 21.044s 0:29.25 317.4%  0+0k 0+0io 0pf+0w
//   76.802u 17.270s 0:28.93 325.1%  0+0k 0+0io 0pf+0w
//   94.198u 20.097s 0:28.85 396.1%  0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 131072 10 5
//   91.180u 7.961s 0:27.42 361.5%   0+0k 0+0io 0pf+0w
//  119.166u 7.088s 0:28.82 438.0%   0+0k 0+0io 0pf+0w
//   74.517u 18.965s 0:30.20 309.5%  0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 131072 10 6
//   91.051u 12.159s 0:29.21 353.3%  0+0k 0+0io 0pf+0w
//   99.874u 14.848s 0:29.52 388.5%  0+0k 0+0io 0pf+0w
//   84.630u 15.556s 0:29.34 341.4%  0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 131072 10 7
//  88.621u 15.375s 0:30.16 344.7%  0+0k 0+0io 0pf+0w
//  95.222u 13.260s 0:29.19 371.6%  0+0k 0+0io 0pf+0w
//  96.982u 12.674s 0:29.80 367.9%  0+0k 0+0io 0pf+0w



// time pipelinemult_test 20000 262144 10 1
//   157.139u 9.139s 1:13.01 227.7%  0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 262144 10 2
//   204.038u 17.325s 1:00.07 368.4% 0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 262144 10 3
//   252.995u 22.251s 1:08.38 402.5% 0+0k 0+0io 0pf+0w

// time pipelinemult_test 20000 262144 10 4
// 219.227u 42.138s 1:07.93 384.7% 0+0k 0+0io 0pf+0

// time pipelinemult_test 20000 262144 10 5
// 296.186u 32.071s 1:09.11 474.9% 0+0k 0+0io 0pf+0w



// time pipelinemult_test  20000 524288 10 1
//   333.371u 7.110s 2:33.66 221.5%  0+0k 0+0io 0pf+0w

// time pipelinemult_test  20000 524288 10 2
//   514.101u 4.286s 2:23.51 361.2%  0+0k 0+0io 0pf+0w

// time pipelinemult_test  20000 524288 10 3
//   526.684u 55.389s 2:33.52 379.1% 0+0k 0+0io 0pf+0w

// time pipelinemult_test  20000 524288 10 4
//   


int main (int argc, char**argv)
{
  if (argc!=4 && argc!=5) {
    cerr << "usage:" << argv[0] << " iterations packetsize multiplier [threads]" << endl;
    exit(1);
  }
  int_8 iterations  = atoi(argv[1]);
  int_8 length  = atoi(argv[2]);
  int_8 multiplier = atoi(argv[3]);
  int_8 threads = (argc==5) ? atoi(argv[4]) : 2;

  // create
  Constant c("constant");
  c.set("iter", iterations);
  c.set("DataLength", length);

  PipelineMult m("pipelineMult");
  m.set("iter", iterations);
  m.set("Multiplier", multiplier);
  m.set("Threads", threads);

  MaxMin  mm("maxmin");
  //Empty  mm("empty");
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
