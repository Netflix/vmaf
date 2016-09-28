#ifndef OC_WORKERCOORDINATORT_H_

// The WorkerCoordinatorT is used to manage (synchronize, create,
// remove) worker threads.  It manages workers that inherit from
// SynchronizedWorker.  (Thus, the templatized class of
// WorkerCoordinatorT is the type of Worker that inherits from
// SynchronizedWorker).  

// ///////////////////////////////////////////// Include Files

#include "ocsynchronizedworker.h"
#include "ocarray.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// The WorkerCoordinatorT Class

template <class T>
class WorkerCoordinatorT {

    // Because we need access to the data members of
    // SynchronizedWorker: the synchronization data
    friend class SynchronizedWorker;

  public:
    
    // ///// Methods

    // Constructor. Initially, no workers: you have call
    // appendNewWorker for every new worker you want to add.
    WorkerCoordinatorT () { }

    // Destructor.  Cleans up all the workers properly and shuts them
    // down.
    OC_INLINE virtual ~WorkerCoordinatorT ();

    // Destroy all workers so that we can change the number of
    // workers.  This makes sure the worker threads shutdown correctly
    // and without problems.

    // Preconditions:  The workers are resting and ready to go
    // Postconditions:  All workers are gone
    OC_INLINE void removeAllWorkers ();

    // Add a new worker:  This adopts the given worker.
    void addNewWorker (T* new_worker) { workerThreads_.append(new_worker); }

    // Give the workers work, have them do it and return from
    // this routine when the work is done.
    // 
    // Preconditions:  The workers are resting and ready to go
    // Post conditions:  All The workers have finished their work,
    //                   and are ready to go again
    OC_INLINE void startAndSynchronizeAllWorkers ();

    // Inspector: Return the worker with unique id.  This method is
    // needed so we can "assignData" to the given worker before we
    // "startAndSynchronizeAllWorkers".
    T& worker (int id) { return *workerThreads_[id]; }

    // Inspector: The number of workers. Note that each worker has a
    // unique id between 0 and number_of_workers-1.
    int_u4 workers() { return workerThreads_.length(); }
    
  protected:

    // ///// Data Members

    // We contain all the worker threads to do our work.  The type "T"
    // is some class that inherits from SynchronizedWorker and is a
    // specialized type of worker for the task to be set before it.
    ArrayPtr<T> workerThreads_;    

    // ///// Methods

    // Disallow copy construction and operator= 
    WorkerCoordinatorT (const WorkerCoordinatorT<T>& c);
    WorkerCoordinatorT& operator= (const WorkerCoordinatorT<T>&);

}; // WorkerCoordinator


OC_END_NAMESPACE


// The implementation: can be put into a .o if you don't want
// everything inlined.

// Currently only support inclusion model for Worker Coordinator:
// splitting into two different types (INCLUSION_MODEL and
// FACTOR_INTO_H_AND_CC is confusing and doesn't work).

#  include "ocworkercoordinatort.cc"



#define OC_WORKERCOORDINATORT_H_
#endif // OC_WORKERCOORDINATORT_H_

