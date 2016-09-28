#ifndef OC_THREADEDPIPELINET_H_

// These classes are used by algorithms to pipeline data: Input data
// packets (typically packets) are taken in, handed off to a thread,
// and reassembled (in the correct order) on output.

// Basic usage: The idea is to give your data (usually packets) to the
// ThreadedPipelineT via the "input" method.  When every thread has a
// piece of data to work on, the pipeline is full, at which point it
// is okay to ask for results via the "output" method.  (You can ask
// for data earlier, but then the pipeline isn't full and you aren't
// taking full advantage of all threads).
//
// Psuedo code:
//--------------------
// ThreadedPipelineT t;
// /* Wait for the pipeline to be primed */
// while (! t.full()) {  
//   get input i
//   t.input(i)
// }
// /* The pipeline is primed, now we can start asking for results */
// while (t.full() && still input data) {
//   output o = t.output();
//   do something with the output
//   get input i
//   if (still input data) t.input(i)
// }
// /* Out of input, running down the pipe:  Works even if pipe never primed */
// while (! t.empty()) {
//   output o = t.output();
// }
// /* All done, pipeline flushed */

// ///////////////////////////////////////////// Includes

#include "ocsynchronizedworker.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// Forwards

template <class InputType, class OutputType>
class ThreadedPipelineT;


// /////////////////////////////////// The ThreadedPipelineTHelper_ Class

// This is a helper class for workers in the threaded pipeline: The
// main thing that a worker wants to satisfy is the work_ method.
// This is where all the work for a pipeline thread gets done.  Note
// that this paradigm assumes that both InputType and OutputType (1)
// have default constructors and (2) support operator= for copying
// input/output to/from the data members and (3) copy constructors
// (for when data is copied in/out)

template <class InputType, class OutputType>
class ThreadedPipelineTHelper_ : public SynchronizedWorker {

    friend class ThreadedPipelineT<InputType, OutputType>;

  public:

    // ///// Methods

    // Constructor.  The creator and thread_number are more overhead
    // stuff.
    OC_INLINE ThreadedPipelineTHelper_ (ThreadedPipelineT<InputType, OutputType>& creator, int_2 thread_number, bool shared_across_processed=false);
    
    // Destructor
    OC_INLINE virtual ~ThreadedPipelineTHelper_ ();

    // Tell the thread to stop: When this returns, we know that it is
    // all done and the resources for that thread can be destructed.
    OC_INLINE void stop ();

  protected:
    
    // ///// Data Members

    // The coordinator that created this: A reference so we can refer
    // to data members of the creator if we have to.
    ThreadedPipelineT<InputType, OutputType>& creator_;

    // The order in which the threads were created.  Thread number 0
    // is the first one created: This is useful if we need to find
    // ourselves in creator_'s array of helpers_
    int_2 threadNumber_;
 
    // Where we put the input to work on:  This is copied with op=
    InputType input_;

    // The final result:  The output method fills this out. Again,
    // this is copied into with op=.    
    OutputType output_;

    // Have we been given an input?  When we start up, all workers are
    // NOT working on something.  When the pipeline is primed (full),
    // every worker is working on something.  As we run down, each
    // worker slowly winds down until no one is working on anything.
    // NOTE: This varaiables is ONLY set/get by the co-ordinator,
    // so there is no reason for any locking.
    bool workingOnSomething_;
    
    // ///// Methods

    // Satisfy SynchronizedWorker: The SynchronizedWorker takes care
    // of a lot of initialization and synchronization for us.
    OC_INLINE virtual void dispatchWork_ ();

    // Each helper needs to override this method to do the actual
    // work.
    virtual OutputType work_ (const InputType& input) = 0;


}; // ThreadedPipelineTHelper_



// ///////////////////////////////////////////// The ThreadedPipelineT Class 

// A class used by algorithms to pipeline data: input packets are
// taken in, handed off to a thread, and reassembled on output.

// The maximum number of thread helpers: An arbitrarily large number
// to prevent accidental creation of very large pools of threads.

#define M2_THREADED_PIPELINE_CEILING 4096

template <class InputType, class OutputType>
class ThreadedPipelineT {

    friend class ThreadedPipelineTHelper_<InputType, OutputType>;

  public:

    // ///// Methods

    // Default constructor: Indicate the number of threads (helpers)
    // we will be using.  If you specify too many (more than
    // M2_THREADED_PIPELINE_CEILING) or a negative number, a
    // MidasException will be thrown indicating the pipeline could not
    // be constructed.
    OC_INLINE ThreadedPipelineT (int_2 number_of_threads = 1);

    // Destructor
    OC_INLINE virtual ~ThreadedPipelineT ();

    // Manipulators

    // Give me an output once we know something is in the pipeline.
    // Precondition:  empty() returns false.
    OC_INLINE OutputType output ();

    // Put next piece of data into the pipeline.
    // Precondition: full() returns false
    OC_INLINE void input (const InputType& input_thing);

    // Returns true if the entire pipeline is full (i.e., all the
    // workers are working away).
    OC_INLINE bool full () const;

    // All workers are done and accounted for: In other words, there
    // is no outstanding work in the pipeline.
    OC_INLINE bool empty () const;

    // Allow us access to any of the helpers.  We return a reference
    // to the pointer because of the rare case when we may need to
    // fill the helper array external to the ThreadedPipelineT
    // (remember friendship isn't inherited, and sometimes its useful
    // to have have access to the array)
    ThreadedPipelineTHelper_<InputType, OutputType>*& helper (int_2 index) 
    { return helpers_[index]; }

    // Inspector to tell us how many helpers (threads) are running.
    int_2 helpers () const { return helpers_.length(); }

  protected:

    // ///// Data Members

    // An array of "helpers", each with their own thread: As input
    // comes in, we hand off the input (assuming the pipe is already
    // primed) to a helper and reassemble the output as if it were
    // done by one thread.
    Array< ThreadedPipelineTHelper_< InputType, OutputType > * > helpers_;

    // The index of the worker we will "wait on" to get the next output.
    int_2 workerWaitingFor_;
    
    // The index of the worker who will get the next work
    int_2 nextWorkerForAssignment_;

}; // ThreadedPipelineT


OC_END_NAMESPACE


// Currently only support inclusion model
#  include "octhreadedpipelinet.cc"


#define OC_THREADEDPIPELINET_H_
#endif // OC_THREADEDPIPELINET_H_
