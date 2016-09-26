
#include "components.h"
#include "ocworkercoordinatort.h"
#include "ocsynchronizedworker.h"
#include "fftw3.h"

// Do "barrier sync" so we divide a packet up into pieces, and have
// workers come together.  This gives some parallelism and limits
// latency.  (Compare to pipelining which gives better parallelism but
// latency suffers).

// If the packets are big enough, we can see some improvement,
// but the packets have to be big enough and do enough work



// -O4 queue size of 4 soc-dualquad1

///////////// COmpare against generation and simple ffts
//////// time baseline_test  200 4194304    #  Generation time
// 9.199u 5.733s 0:10.84 137.6%    0+0k 0+0io 0pf+0w
///////// time ./fft_test 200 4194304 524288
//   76.488u 9.520s 1:04.92 132.4%   0+0k 0+0io 1pf+0w

// time ./barrierfft_test 200 4194304 524288 1
//  65.335u 9.144s 0:53.96 138.0%   0+0k 0+0io 0pf+0w

// time ./barrierfft_test 200 4194304 524288 2
//  71.972u 10.304s 0:35.55 231.4%  0+0k 0+0io 0pf+0w

// time ./barrierfft_test 200 4194304 524288 3
//  82.602u 9.413s 0:38.63 238.1%   0+0k 0+0io 0pf+0w

// time ./barrierfft_test 200 4194304 524288 4  # Work divides EVENLY,so better
//  85.822u 10.823s 0:27.27 354.3%  0+0k 0+0io 0pf+0w

// time ./barrierfft_test 200 4194304 524288 5
//  97.654u 11.938s 0:34.01 322.1%  0+0k 0+0io 0pf+0w


// time ./barrierfft_test 200 4194304 524288 6
//  124.715u 10.857s 0:36.67 369.6% 0+0k 0+0io 0pf+0w
//  120.371u 12.660s 0:36.39 365.5% 0+0k 0+0io 0pf+0w

// time ./barrierfft_test 200 4194304 524288 7
//  154.622u 11.522s 0:33.16 501.0% 0+0k 0+0io 0pf+0w

// time ./barrierfft_test 200 4194304 524288 8
//  151.388u 13.392s 0:30.66 537.4% 0+0k 0+0io 0pf+0w
//  150.800u 11.972s 0:30.21 538.7% 0+0k 0+0io 0pf+0w



// Have separate threads work on different regions of a packet
struct FFTPacketWorker : public SynchronizedWorker {
  FFTPacketWorker (int worker_number, fftwf_plan& plan,
		   StreamingPool* sp) :
    SynchronizedWorker("FFTPacketWorker"+Stringize(worker_number)),
    plan_(plan),
    region_(0),
    FFTLength_(0),
    numberOfFFTs_(0),
    sp_(sp)
  { }

  // Manager calls directly to set up worker 
  void assignWork (fftwf_complex* region, int fft_len, int number_of_ffts)
  {
    region_ = region;
    FFTLength_ = fft_len;
    numberOfFFTs_ = number_of_ffts;
    // cerr << "region_ = " << region_ << " FFTLength_ = " << FFTLength_ << " numberOfFFTS_ = " << numberOfFFTs_ << endl;
  }

protected:
  
  // Called for you by framework when work is ready
  virtual void dispatchWork_ ()
  {
    const int len = numberOfFFTs_;
    fftwf_complex* data = region_;

    for (int ii=0; ii<len; ii++) {
      //cerr << "DATA" << data << " ii=" << ii << endl;
      //complex_8* pr = (complex_8*)data;
      //cerr << "BEFORE FFT" << endl;
      //for (int jj=0; jj<FFTLength_; jj++) {
      //cerr << pr[jj] << " ";
      //}
      //cerr << endl;
      fftwf_execute_dft(plan_, data, data); // inplace FFT
      //cerr << "AFTER FFT" << endl;
      //for (int jj=0; jj<FFTLength_; jj++) {
      //cerr << pr[jj] << " ";
      //}
      //cerr << endl;
      //complex_8* a = (complex_8*) other;
      //for (int jj=0; jj<FFTLength_; jj++) {
      //cerr << a[jj] << " ";
      // }
      //cerr << endl;
      data += FFTLength_;
    }
  }

  fftwf_plan& plan_;
  fftwf_complex* region_;  // Data to work on, NOT adopted
  int FFTLength_;
  int numberOfFFTs_;         // length of data
  StreamingPool* sp_;

}; // PacketWorker


// Spend multiple threads across a packet
class ThreadFFT : public Transformer, 
		  public WorkerCoordinatorT<FFTPacketWorker> {
public:
  ThreadFFT (const string& name) : 
    Transformer(name),
    WorkerCoordinatorT<FFTPacketWorker>(),
    threads_(-1),
    fftSize_(0),
    plan_(0),    // empty plan to start
    planIn_(0),
    planOut_(0),
    sp_(0)
  {
    // Have to use memory that is 16-byte aligned for FFTW SSE and SSE2 instr
    const int len = 1024*1024*256;
    char* buffer = new char[len];
    sp_ = StreamingPool::CreateStreamingPool(buffer, len, 16); // SSE and SSE2 
  }
  
  void initThreads ()
  {
    if (threads_==-1) {
      // Create a plan for the threads to share
      int fft_size = get("FFTSize");
      planIn_  = (fftwf_complex*)sp_->allocate(sizeof(complex_8)*fft_size);
      planOut_ = planIn_;  // (fftwf_complex*)sp_->allocate(sizeof(complex_8)*fft_size);
      fftSize_ = fft_size;
      plan_ = fftwf_plan_dft_1d(fftSize_, planIn_, planOut_, FFTW_FORWARD,
				FFTW_ESTIMATE|FFTW_DESTROY_INPUT);
    
      // Now, have all the threads share the plan
      int threads = get("Threads", 1);
      cerr << "Threads: Looking at " << threads << endl;
      if (threads<1) throw logic_error("Have to have at least one thread");

      for (int ii=0; ii<threads; ii++) {
	addNewWorker(new FFTPacketWorker(ii, plan_, sp_));
      }
      threads_ = threads;
    }
  }
  
  virtual bool transform (const Val& in, Val& out)
  { 
    initThreads();

    // Unravel packet
    Proxy out_array;
    {
      TransactionLock tl(in);
      Array<complex_8>& in_array = in("DATA");
      out_array = Shared(sp_, in_array); // copies data out as 16-byte aligned 
    }
       
    // Do work: make sure use StreamingPool to allocate data
    Array<complex_8>& a = out_array;
    // cerr << "DATA BEFORE THREADS" << a << endl;
    complex_8* data = a.data();

    // Assign all the threads their "region" to work on
    int regions = a.length() / fftSize_;
    int threads = workers();
    const int length = regions / threads; 
    const int remain = regions % threads;
    fftwf_complex* region_start = (fftwf_complex*)data;
    for (int ii=0; ii<threads; ii++) {
      worker(ii).assignWork(region_start, fftSize_, length);
      if (ii==threads-1) {  // last one has potentially more data
          worker(ii).assignWork(region_start, fftSize_, length+remain);
      }
      region_start += length*fftSize_;
    }
    startAndSynchronizeAllWorkers();

    // All done
    out = Locked(new Tab("{'HEADER':{}, 'DATA':None}"));
    out["DATA"] = out_array;

    return true; 
  }
  
protected:
  int threads_;
  int fftSize_;
  fftwf_plan plan_;
  fftwf_complex *planIn_, *planOut_; // Only used for plan creation
  StreamingPool* sp_;
  
}; // ThreadedMult

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

  ThreadFFT m("threadedMult");
  m.set("iter", iterations);
  m.set("FFTSize", fftsize);
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
