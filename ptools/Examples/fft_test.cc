
#include "components.h"
#include "fftw3.h"
#include "ocstreamingpool.h"

// Baseline for how fast we can ship FFTs through a system

// time ./fft_test 200 4194304 524288
//   76.488u 9.520s 1:04.92 132.4%   0+0k 0+0io 1pf+0w

// time ./fft_test 200 4194304 4194304
//   141.261u 9.847s 2:07.85 118.1%  0+0k 0+0io 0pf+0w
   

// A simple FFT algorithm that reads the input and computes 
class FFTAlg : public Transformer {
public:
  FFTAlg (const string& name) : 
    Transformer(name),
    sp_(0),
    fftSize_(-1),
    planIn_(0), 
    planOut_(0),
    plan_(0)
  {
    const int len = 1024*1024*256;
    char* buffer = new char[len];
    sp_ = StreamingPool::CreateStreamingPool(buffer, len, 16); // SSE and SSE2 

    // Default FFTSize
    set("FFTSize", 1024);
    FFTInit();
  }

  virtual ~FFTAlg () 
  {
    if (plan_) {
      fftwf_destroy_plan(plan_);
      plan_ = NULL;
    }
    sp_->scheduleForDeletion();
    char* s = (char*)sp_;
    delete [] s;
  }

  // Create a new plan (if necessary) for the current FFTSize
  int FFTInit ()
  {
    // Have we seen this kind of FFT before?
    int fft_size = get("FFTSize");
    if (fftSize_!=fft_size) {
      if (plan_) {
        fftwf_destroy_plan(plan_);
        plan_ = NULL;
      }
      if (planIn_) sp_->deallocate((char*)planIn_);
      planIn_ = (fftwf_complex*)sp_->allocate(sizeof(complex_8)*fft_size);
      if (planOut_) sp_->deallocate((char*)planOut_);
      planOut_ = (fftwf_complex*)sp_->allocate(sizeof(complex_8)*fft_size);

      fftSize_ = fft_size;
      plan_ = fftwf_plan_dft_1d(fftSize_, planIn_, planOut_, FFTW_FORWARD,
				FFTW_ESTIMATE|FFTW_DESTROY_INPUT);
    }
    return fft_size;
  }

  virtual bool transform (const Val& in, Val& out)
  { 
    // Check and see if we need to create a new plan
    int fft_size = FFTInit();

    // Unravel packet
    Proxy out_array;
    {
      Val packet = in;
      TransactionLock tl(in);
      Array<complex_8>& in_array = packet("DATA");
      out_array = Shared(sp_, in_array); // copies input
    }
       
    // Do work: make sure use StreamingPool to allocate data
    Array<complex_8>& a = out_array;

    // Go through and do all the FFTS!
    fftwf_complex* data = (fftwf_complex*)a.data();
    const int howmany = a.length() / fft_size;
    const int remain = a.length() % fft_size;
    if (remain) { cerr << "Warning: dropping some input" << endl; }
    for (int ii=0; ii<howmany; ii++) {
      fftwf_execute_dft(plan_, data, data); // inplace
      data += fft_size;
    }

    // All done
    out = in;

    return true;
  }
  
protected:
  StreamingPool* sp_;
  int fftSize_;
  fftwf_complex *planIn_, *planOut_;
  fftwf_plan plan_;
  
}; // FFTAlg

int main (int argc, char**argv)
{
  if (argc!=4) {
    cerr << "usage:" << argv[0] << " iterations packetsize multiplier" << endl;
    exit(1);
  }
  int_8 iterations  = atoi(argv[1]);
  int_8 length  = atoi(argv[2]);
  int_8 fft_size = atoi(argv[3]);

  // create
  Constant c("constant");
  c.set("iter", iterations);
  c.set("DataLength", length);

  FFTAlg m("fftalg");
  m.set("iter", iterations);
  m.set("FFTSize", fft_size);

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
