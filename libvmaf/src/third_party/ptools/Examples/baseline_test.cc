
#include "components.h"


// time baseline_test 20000 65536
//  13.848u 0.789s 0:08.15 179.3%   0+0k 0+0io 0pf+0w
//  14.152u 0.248s 0:07.91 181.9%   0+0k 0+0io 0pf+0w
//  14.126u 0.246s 0:07.93 181.0%   0+0k 0+0io 0pf+0w

// time ./baseline_test 200 4194304
//  9.191u 5.654s 0:10.75 138.0%    0+0k 0+0io 0pf+0w
//  9.218u 5.046s 0:10.18 139.9%    0+0k 0+0io 0pf+0w

int main (int argc, char**argv)
{
  if (argc!=3) {
    cerr << "usage:" << argv[0] << " iterations packetsize " << endl;
    exit(1);
  }
  int_8 iterations  = atoi(argv[1]);
  int_8 length  = atoi(argv[2]);

  // create
  Constant c("constant");
  c.set("iter", iterations);
  c.set("DataLength", length);

  MaxMin  mm("maxmin");
  mm.set("iter", iterations);

  // connect
  CQ* a = new CQ(4);
  c.connect(0, a);
  mm.connect(a, 0);

  // start
  mm.start();
  c.start();

  // Wait for everyone to finish
  mm.wait(); 
  c.wait();  

  cout << "Max = " << mm.get("Max") << endl;
  cout << "Min = " << mm.get("Min") << endl;

  delete a;
}
