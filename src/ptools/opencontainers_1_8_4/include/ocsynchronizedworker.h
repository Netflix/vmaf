#ifndef OC_SYNCHRONIZEDWORKER_H_

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "octhread.h"
#include "ocsynchronizer.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// The SynchronizedWorker Class

// The SynchronizedWorker class exists to do a "piece of work":  Multiple
// SynchronizedWorkers will co-operate and do a larger piece of work.
// The class WorkerCoordinatorT is the class that synchronizes
// all workers.   

// To make a SynchronizedWorker for a specific task, inherit from
// SynchronizedWorker and override dispatchWork_ (that's where
// the computation will occur for this worker).  

class SynchronizedWorker : public OCThread {

    // This should really be protected, and we should have "template
    // <class T> friend class WorkerCoordinatorT;" but, that only
    // works with newer compilers.  Someday, when all compilers
    // support this feature, we will limit access properly, but right
    // now, it has to be public so everyone can see inside.  It's best
    // to think of the data members as protected (so they have the _
    // at the end of their names).

  public:  // protected!  See comment above


    // ///// Data Members

    // A flag to indicate that we are done working.  This is modified
    // outside a mutex region, so mark it volatile.  NB: This should
    // ONLY by modified by the WorkerCoordinatorT, since that needs
    // to know that the worker is going away, so it doesn't try to
    // wait for a signal from a deceased thread.
    volatile bool doneExecuting_;

    // Used for synchronization.  This is the first barrier ...  all
    // workers are waiting for "work to be ready" ...
    CondVar       workReady_;

    // The second memory barrier.  All workers have done their work,
    // and the co-ordinating thread is waiting to be notified that all
    // threads have reached this barrier so that it knows the current
    // batch of work is done (and has to reassign new work)
    CondVar         barrierReached_;


    // ///// Methods

    // Construct the SynchronizedWorker.  Note that someone inheriting
    // from this class might construct this with more information, but
    // that's the whole purpose: Someone inherits from this class to
    // fill in what's missing.
    OC_INLINE SynchronizedWorker (const string& name,
				  bool shared_across_processes=false,
				  bool start_upon_construction=true);

    // Destructor
    virtual ~SynchronizedWorker () { }

    // Assertion: this worker at the top of the loop, waiting to
    // get told what to do.  Start him up.
    OC_INLINE void startUp ();

    // Assertion: this worker is working, or done. Wait for him
    // to finish, and return when he is ready for the next round.
    OC_INLINE void waitFor ();

    // Overide the M2Thread process.  This is where synchronization
    // and work happens.  The user DOES NOT intercept this method:
    // This is handled for him ... the user should only intercept
    // dispatchWork_ (below).
    OC_INLINE virtual void process_ ();

    // Dispatch the "real work".  All the synchronization is taken
    // care of outside of this method.  When you enter this method,
    // the worker does the work he has been assigned to do and when
    // you leave this method the work of this thread is done until it
    // is called again.
    virtual void dispatchWork_ () = 0;

    // When someone inherits from this class, they will need to have
    // some mutators to change the data that comes in after every
    // dispatchWork_. Since those will be VERY context specific, we
    // leave it to the user to supply the arguments and name of the
    // routine ... we suggest "assignData"

    // assignData (supply context specific data here)
    
}; // SynchronizedWorker



// The Thread Main Loop
inline static void* SyncWorkerMainLoop (void* data) 
{
  SynchronizedWorker* sw = (SynchronizedWorker*)data;
  sw->process_();
  return NULL;
}


OC_END_NAMESPACE

// The implementation: can be put into a .o if you don't want
// everything inlined.
#if !defined(OC_FACTOR_INTO_H_AND_CC)
#  include "ocsynchronizedworker.cc"
#endif 


#define OC_SYNCHRONIZEDWORKER_H_
#endif // OC_SYNCHRONIZEDWORKER_H_
