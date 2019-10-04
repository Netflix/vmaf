
#include "pipelinetransformer.h"

// One thread per packet: Packets come in and are assigned in order
struct MultPipelineWorker : public ThreadedPipelineTHelper_<Val,Val> {

  // Constructor: Give each element of the pipeline the resources needed
  MultPipelineWorker (ThreadedPipelineT<Val,Val>& god, int worker_number,
		      real_8 multiplier):
    ThreadedPipelineTHelper_<Val, Val>(god, worker_number),
    multiplier_(multiplier),
    sp_(0)
  { 
    // Each worker should have its own pool so there's little
    // contention for the memory allocation mutex.
    const int len = 1024*1024*1024;
    char* buffer = new char[len];
    sp_ = StreamingPool::CreateStreamingPool(buffer, len, 16); // SSE and SSE2 
  }

  virtual ~MultPipelineWorker ()
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
    int len = a.length();
    real_4 mult = multiplier_;

    for (int ii=0; ii<len; ii++) {
      data[ii] *= mult;
    }
    
    // All done
    Val out = Locked(new Tab("{'HEADER':{}, 'DATA':None}"));
    out["DATA"] = out_array;
    //cerr << "after ffts" << out << endl;
    return out;
  }

  // ////// Data Members
  StreamingPool* sp_;  // make sure each has own pool
  real_8 multiplier_;
  
}; // FFTPipelineWorker


// One thread per input packet:  this works because each input is
// separate from the previous
class PipelineMult : public PipelineTransformer { 

public:
  PipelineMult (const string& name) : 
    PipelineTransformer(name),
    threads_(-1)
  { }

  void init ()
  {
    if (threads_==-1) {    
      // Figure out number of threads
      int threads = get("Threads", 1);
      real_8 mult = get("Multiplier", 10);

      cerr << "Threads: Looking at " << threads << endl;
      if (threads<1) throw logic_error("Have to have at least one thread");

      // Create a new pipeline
      pipeline_ = new ThreadedPipelineT<Val, Val>(threads);
      for (int ii=0; ii<threads; ii++) {
	pipeline_->helper(ii) = 
	  new MultPipelineWorker(*pipeline_, ii, mult);
      }
      threads_ = threads;
    }
  }
    
protected:
  int threads_;
  
}; // PipelineMult


