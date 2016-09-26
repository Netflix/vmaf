
#include "pipelinetransformer.h"
#include "fftw3.h"

void LoadWisdom ()
{
  FILE* fp = fopen("./WISDOM", "r");
  if (fp==0) return;
  cout << "Loading Wisdom ..." << endl;
  fftwf_import_wisdom_from_file(fp);
  fclose(fp);
  cout << " ... done loading WISDOM" << endl;
}

void SaveWisdom ()
{
  FILE* fp = fopen("./WISDOM", "w");
  fftwf_export_wisdom_to_file(fp);
  fclose(fp);
}


// One thread per packet: Packets come in and are assigned in order
struct FFTPipelineWorker : public ThreadedPipelineTHelper_<Val,Val> {

  // Constructor: Give each element of the pipeline the resources needed
  FFTPipelineWorker (ThreadedPipelineT<Val,Val>& god, int worker_number, 
		     fftwf_plan& plan, int fft_len) :
    ThreadedPipelineTHelper_<Val, Val>(god, worker_number),
    plan_(plan),
    FFTSize_(fft_len),
    sp_(0)
  { 
    // Each worker should have its own pool so there's little
    // contention for the memory allocation mutex.

    // Have to use memory that is 16-byte aligned for FFTW SSE and SSE2 instr
    const int len = 1024*1024*256;
    char* buffer = new char[len];
    sp_ = StreamingPool::CreateStreamingPool(buffer, len, 16); // SSE and SSE2 
  }

  virtual ~FFTPipelineWorker ()
  {
    sp_->scheduleForDeletion();
  }

protected:
  
  // Called when a new packet given to my stage of the pipeline
  Val work_ (const Val& input)
  {
    // Unravel packet
    Proxy out_array;
    {
      TransactionLock tl(input);
      Array<complex_8>& in_array = input("DATA");
      out_array = Shared(sp_, in_array); // copies data out as 16-byte aligned 
    }
    
    // Do work: make sure use StreamingPool to allocate data
    Array<complex_8>& a = out_array;
    complex_8* data = a.data();
    //cerr << "before fft" << a << endl;

    int ffts = a.length()/FFTSize_;
    if (a.length() % FFTSize_) {
      cerr << "Warning: packets aren't right size" << endl;
    }
    for (int ii=0; ii<ffts; ii++) {
      fftwf_execute_dft(plan_, (fftwf_complex*)data, (fftwf_complex*)data); 
      data += FFTSize_;
    }
    
    // All done
    Val out = Locked(new Tab("{'HEADER':{}, 'DATA':None}"));
    out["DATA"] = out_array;
    //cerr << "after ffts" << out << endl;
    return out;
  }

  // ////// Data Members
  fftwf_plan& plan_;   // Each fft needs a plan
  int FFTSize_;        // keep the size handy
  StreamingPool* sp_;  // make sure arrays are 16 byte aligned 

}; // FFTPipelineWorker


// One thread per input packet:  this works because each input is
// separate from the previous
class PipelineFFT : public PipelineTransformer { 

public:
  PipelineFFT (const string& name) : 
    PipelineTransformer(name),
    threads_(-1),
    fftSize_(0),
    plan_(0),    // empty plan to start
    planIn_(0),
    planOut_(0),
    sp_(0)
  {
    // Have to use memory that is 16-byte aligned for FFTW SSE and SSE2 instr
    const int len = 1024*1024*64;
    char* buffer = new char[len];
    sp_ = StreamingPool::CreateStreamingPool(buffer, len, 16); // SSE and SSE2 
  }

  virtual ~PipelineFFT()
  {
    if (planIn_) sp_->deallocate((char*)planIn_);
    if (plan_) fftwf_destroy_plan(plan_);
    sp_->scheduleForDeletion();
  }

  virtual void init ()
  {
    if (threads_==-1) {
      // Create a plan for the threads to share
      LoadWisdom();

      int fft_size = get("FFTSize");
      planIn_  = (fftwf_complex*)sp_->allocate(sizeof(complex_8)*fft_size);
      planOut_ = planIn_;  // (fftwf_complex*)sp_->allocate(8*fft_size);
      fftSize_ = fft_size;
      plan_ = fftwf_plan_dft_1d(fftSize_, planIn_, planOut_, FFTW_FORWARD,
				FFTW_ESTIMATE|FFTW_DESTROY_INPUT);
      //FFTW_EXHAUSTIVE|FFTW_DESTROY_INPUT);
    
      SaveWisdom();

      // Now, have all the threads share the plan
      int threads = get("Threads", 1);
      cerr << "Threads: Looking at " << threads << endl;
      if (threads<1) throw logic_error("Have to have at least one thread");

      // Create a new pipeline
      pipeline_ = new ThreadedPipelineT<Val, Val>(threads);
      for (int ii=0; ii<threads; ii++) {
	pipeline_->helper(ii) = 
	  new FFTPipelineWorker(*pipeline_, ii, plan_, fftSize_);
      }
      threads_ = threads;
    }
  }
  
protected:
  int threads_;
  int fftSize_;
  fftwf_plan plan_;
  fftwf_complex *planIn_, *planOut_; // Only used for plan creation
  StreamingPool* sp_;
  
}; // PipelineFFT


