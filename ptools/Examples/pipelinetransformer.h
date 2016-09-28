
#include "components.h"
#include "octhreadedpipelinet.h"

// One thread per input packet: these work because each input is
// separate from the previous
class PipelineTransformer : public Transformer { 

public:
  PipelineTransformer (const string& name) : 
    Transformer(name),
    pipeline_(0)
  { }

  virtual ~PipelineTransformer ()
  {
    delete pipeline_;
  }

  // Called at the top of every transform to make sure 
  // everything initialized okay
  virtual void init () = 0;
  
  // This is where the magic happens: all pipeline transformers works
  // the same way.
  virtual bool transform (const Val& in, Val& out)
  { 
    init(); // Make sure everything set up before processing

    // Prime pipeline if not enough data yet
    if (! pipeline_->full()) { 
      pipeline_->input(in);
      return false; // Not enough data yet
    }
    
    // Assertion: The pipeline is now primed, now we can start asking
    // for results
    out = pipeline_->output();
    pipeline_->input(in);

    return true;
  }

protected:
  // When running down, get last few packets out
  virtual void rundown_ ()
  {
    // See if last packet
    int iter = get("iter");
    if (iter==final_) {
      while (! pipeline_->empty()) {
	Val out = pipeline_->output();
	output_->enq(out);
      }
    }
  }
  
protected:

  // The pipeline to fill up: Pointer so we can set threads at
  // initialization instead on at construction time
  ThreadedPipelineT<Val, Val>* pipeline_; 
  
}; // PipelineTransformer


