
#include "components.h"
#include "ocworkercoordinatort.h"
#include "ocsynchronizedworker.h"

// Even with huge packets, still no much speedup because NOT ENOUGH WORK!

///////////// baseline_test  200 4194304    #  Generation time
// 9.199u 5.733s 0:10.84 137.6%    0+0k 0+0io 0pf+0w

// -O4 queu size of 4 soc-dualquad1
// time ./barriermult_test 200 4194304 10 1
//   23.503u 15.853s 0:18.54 212.2%  0+0k 0+0io 0pf+0w
//   23.389u 16.033s 0:18.59 211.9%  0+0k 0+0io 0pf+0w

// time ./barriermult_test 200 4194304 10 2
//   27.226u 16.405s 0:18.31 238.2%  0+0k 0+0io 0pf+0w

// time ./barriermult_test 200 4194304 10 4
//  30.031u 15.329s 0:18.53 244.7%  0+0k 0+0io 0pf+0w

// time ./barriermult_test 200 4194304 10 8
//  37.437u 15.892s 0:18.12 294.2%  0+0k 0+0io 0pf+0w



// Have separate threads work on different regions of a packet
struct PacketWorker : public SynchronizedWorker {
  PacketWorker (int worker_number) :
    SynchronizedWorker("PacketWorker"+Stringize(worker_number)),
    region_(0),
    length_(0)
  { }

  // Manager calls directly to set up worker 
  void assignWork (complex_8* region, int length)
  {
    region_ = region;
    length_ = length;
  }

protected:
  
  // Called for you by framework when work is ready
  virtual void dispatchWork_ ()
  {
    const int len = length_;
    complex_8* data = region_;
    for (int ii=0; ii<len; ii++) {
      data[ii] *= 10;
    }
  }

  complex_8* region_;  // Data to work on, NOT adopted
  int length_;         // length of data

}; // PacketWorker


// Spend multiple threads across a packet
class ThreadMult : public Transformer, 
		   public WorkerCoordinatorT<PacketWorker> {
public:
  ThreadMult (const string& name) : 
    Transformer(name),
    WorkerCoordinatorT<PacketWorker>(),
    threads_(-1)
  {
    
  }
  
  void initThreads ()
  {
    if (threads_==-1) {
      int threads = get("Threads", 1);
      cerr << "Threads: Looking at " << threads << endl;
      if (threads<1) throw logic_error("Have to have at least one thread");
      for (int ii=0; ii<threads; ii++) {
	addNewWorker(new PacketWorker(ii));
      }
      threads_ = threads;
    }
  }
  
  virtual bool transform (const Val& in, Val& out)
  { 
    initThreads();

    // Unravel packet
    Proxy out_array = Locked(new Array<complex_8>());
    {
      Val packet = in;
      TransactionLock tl(in);
      Array<complex_8>& in_array = packet("DATA");
      Array<complex_8>& oa = out_array;
      oa = in_array; // copies input
    }
       
    // Do work: make sure use StreamingPool to allocate data
    Array<complex_8>& a = out_array;
    complex_8* data = a.data();

    // Assign all the threads their "region" to work on
    int threads = workers();
    const int length = a.length() / threads; 
    const int remain = a.length() % threads;
    complex_8* region_start = data;
    for (int ii=0; ii<threads; ii++) {
      worker(ii).assignWork(region_start, length);
      if (ii==threads-1) {  // last one has potentially more data
          worker(ii).assignWork(region_start, length+remain);
      }
      region_start += length;
    }
    startAndSynchronizeAllWorkers();

    // All done
    out = Locked(new Tab("{'HEADER':{}, 'DATA':None}"));
    out["DATA"] = out_array;

    return true;
  }
  
protected:
  int threads_;
}; // ThreadedMult

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

  ThreadMult m("threadedMult");
  m.set("iter", iterations);
  m.set("Multiplier", multiplier);
  m.set("Threads", threads);

  MaxMin  mm("maxmin");
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
