
// ///////////////////////////////////////////// Includes

// Currently only support template inclusion model
// #  include "octhreadedpipelinet.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// ThreadedPipelineTHelper_ Methods

template <class InputType, class OutputType>
OC_INLINE ThreadedPipelineTHelper_<InputType, OutputType>::
ThreadedPipelineTHelper_ (ThreadedPipelineT<InputType, OutputType>& creator, 
			  int_2 thread_number, bool shared_across_processes):
  SynchronizedWorker("ThreadedPipelineTHelper_"+Stringize(thread_number),
		     shared_across_processes),
  creator_(creator),
  threadNumber_(thread_number),
  workingOnSomething_(false)
{ }


template <class InputType, class OutputType>
OC_INLINE ThreadedPipelineTHelper_<InputType, OutputType>::~ThreadedPipelineTHelper_ ()
{
  stop();
}


template <class InputType, class OutputType>
OC_INLINE void ThreadedPipelineTHelper_<InputType, OutputType>::stop ()
{
  // In case the ThreadedPipelineT is stuck waiting for this helper,
  // the thread running this routine prods the ThreadedHelperT.  This
  // also indicates that this helper thread is to run down.
  doneExecuting_ = true;
  workReady_.lock();
  doneExecuting_ = true;
  workReady_.broadcast();
  workReady_.unlock();

  // Now, wait for the thread resources to evaporate
  // waitup();
  barrierReached_.lock();
  while (! barrierReached_()) {
    barrierReached_.wait();
  }
  barrierReached_() = false;
  barrierReached_.unlock();
}


template <class InputType, class OutputType>
OC_INLINE void ThreadedPipelineTHelper_<InputType, OutputType>::dispatchWork_ ()
{
  output_ = work_(input_);
}

// ///////////////////////////////////////////// ThreadedPipelineT Methods

template <class InputType, class OutputType> 
OC_INLINE ThreadedPipelineT<InputType, OutputType>::ThreadedPipelineT (int_2 number_of_threads) :
  helpers_((number_of_threads<=0||number_of_threads>M2_THREADED_PIPELINE_CEILING) ? 1 : number_of_threads),  // Make sure the helpers_ array is initialized to the right number, but only if the range is valid.
  workerWaitingFor_(0),
  nextWorkerForAssignment_(0)
{
  // Do this check again after all our data memebers are in a known
  // state (not sure we want to be throwing an exception in the middle
  // of constructing the array)
  if (number_of_threads<=0 || number_of_threads > M2_THREADED_PIPELINE_CEILING) {
    throw logic_error("You cannot create a ThreadedPipeline of "+Stringize(number_of_threads));
  }

  // Initialize space for all the helpers
  for (int_2 ii=0; ii<number_of_threads; ii++) {
    helpers_.append(0);
  }
}



template <class InputType, class OutputType>
OC_INLINE ThreadedPipelineT<InputType, OutputType>::~ThreadedPipelineT ()
{
  // Do something to stop everyone. Delete space for all the helpers
  int number_of_threads = helpers();
  for (int_2 ii=0; ii<number_of_threads; ii++) {
    delete helpers_[ii];  // Note that the destructor does a stop
    helpers_[ii] = 0;
  }

}

template <class InputType, class OutputType>
OC_INLINE void ThreadedPipelineT<InputType, OutputType>::input (const InputType& input)
{
  // Notice how this works: We always assume that the least recent
  // worker has already given up his output so that he is always
  // ready to take a new input.  
  ThreadedPipelineTHelper_<InputType,OutputType>& w = 
    *helpers_[nextWorkerForAssignment_];

  // The least recent guy should have already returned his output, so
  // he is ready again to take a new input.
  w.workReady_.lock();

  w.input_ = input;
 
  w.workReady_() = true;
  w.workReady_.signal();
  w.workReady_.unlock();

  // This worker "checks in" with the boss to say he's working on
  // something (since only the "coordinator" thread ever sets or
  // inspects this variable, no need for locking).
  w.workingOnSomething_ = true;


  // Next time input is called, the next helper in the queue becomes
  // the least recent worker
  nextWorkerForAssignment_ = (nextWorkerForAssignment_+1) % helpers_.length(); 
}


template <class InputType, class OutputType>
OC_INLINE bool ThreadedPipelineT<InputType, OutputType>::full () const
{
  ThreadedPipelineTHelper_<InputType,OutputType>& w =
    *helpers_[nextWorkerForAssignment_];
  return w.workingOnSomething_;
}


template <class InputType, class OutputType>
OC_INLINE bool ThreadedPipelineT<InputType, OutputType>::empty () const
{
  ThreadedPipelineTHelper_<InputType,OutputType>& w =
    *helpers_[workerWaitingFor_];
  return ! w.workingOnSomething_;
}
  

template <class InputType, class OutputType>
OC_INLINE OutputType ThreadedPipelineT<InputType, OutputType>::output ()
{
  OutputType all_done;

  // The given worker has the output.  Get his output
  ThreadedPipelineTHelper_<InputType,OutputType>& w =
    *helpers_[workerWaitingFor_];

  // Lock and wait for the work to be done. When it's done, we'll get
  // signalled
  w.barrierReached_.lock();
  while (! w.barrierReached_()) {
    w.barrierReached_.wait();
  }

  all_done = w.output_;

  // Once reached, can reset it and unlock the mutex
  w.barrierReached_() = false;
  w.barrierReached_.unlock();       

  // Indicate worker has "checked in" and is no longer working on
  // anything.  (Only the "coordinator" thread ever sets or checks
  // this, so we don't need to lock it)
  w.workingOnSomething_ = false;

  // Move up to next worker
  workerWaitingFor_ = (workerWaitingFor_ + 1) % helpers_.length();

  return all_done;
}


OC_END_NAMESPACE
