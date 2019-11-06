#ifndef OCTHREAD_H_

// The OCThread class is a simple wrapper around POSIX threading
// calls.  By default, we create the threads with PTHREAD_SCOPE_SYSTEM
// and as detached threads (i.e., you don't have to join).  Threads
// are created as passive until you want something to run, you call
// "start" with a pointer-to-a-function (see ThreadMain typedef
// below) to be the main loop.  We add another wrapper 

#include "ocport.h"
#include <pthread.h>

OC_BEGIN_NAMESPACE

// All thread routines take a routine with this signature.
typedef void * (*ThreadMain)(void *);


// This class encapsulates the POSIX pthread calls
class OCThread {
  public:

    OCThread (const string& thread_name, bool run_detached=true) :
      threadName_(thread_name),
      started_(false),
      detached_(run_detached?PTHREAD_CREATE_DETACHED:PTHREAD_CREATE_JOINABLE)
    {
      pthread_attr_init(&attr_);
      pthread_attr_setscope(&attr_, PTHREAD_SCOPE_SYSTEM);
      pthread_attr_setdetachstate(&attr_, detached_);
    }


    bool started () const { return started_; } 

    void threadName (const string& thread_name) { threadName_ = thread_name; }
    string threadName () const { return threadName_; }

    // Start the thread (if not already started) to run the given main
    // routine with the given data.
    inline void start (ThreadMain start_routine, void* data);

    // If the thread is joinable, wait until it quits and get it's
    // return value.  If the thread is not joinable, this immediately
    // returns with NULL.
    inline void* join ();
  
    // void cancel (); // Be careful when you call this!

    ~OCThread () { if (!detached_) join(); }

  protected:
    pthread_t      tid_;
    pthread_attr_t attr_;
    string         threadName_; 
    bool           started_;
    int            detached_;  // One of PTHREAD_CREATE_JOINABLE, PTHREAD_CREATE_DETACHED

}; // OCThread


// By default, we add another loop around the main routine you pass
// in: this is to make sure we catch a few exceptions and implement
// the mechanism to do joins easily.
struct ThreadMainLoopData {
    ThreadMainLoopData (ThreadMain start, OCThread* id, void* data) :
      start_routine(start), thread_id(id), original_data(data) { }
    ThreadMain start_routine;
    OCThread*  thread_id;
    void*      original_data;
    void*      return_value;
};


inline void* ThreadMainLoop (void* data)
{
  ThreadMainLoopData* tmld = reinterpret_cast<ThreadMainLoopData*>(data);
  void* return_value = 0;
  try {
    return_value = tmld->start_routine(tmld->original_data);
    tmld->return_value = return_value;
  } catch (const exception& e) {
    cerr << "Thread:" << tmld->thread_id->threadName() << " " 
         << e.what() << endl;
  } catch (...) {
    cerr << "Thread:" << tmld->thread_id->threadName() << " " 
         << "Unknown exception caught" << endl;
  }

  pthread_exit(data); // return value is in the structure
  return 0;
}

 
extern "C" { 
  // Used for casting purposes only
  typedef void * (*CThreadMain)(void *);
}

inline void OCThread::start (ThreadMain start_routine, void* data)
{
  if (!started_) {
    ThreadMainLoopData* tmld = new ThreadMainLoopData(start_routine,this,data);

    int res = pthread_create(&tid_, &attr_, (CThreadMain)ThreadMainLoop, tmld);
    if (res!=0) {
      perror("pthread_create");
      throw runtime_error("Couldn't create thread: not enough sytem resources");
    }
  }
  started_ = true;
}

inline void* OCThread::join ()
{
  if (detached_==PTHREAD_CREATE_DETACHED) {
    throw runtime_error("Thread:"+threadName_+"is running detached:"
			"joins do not work.");
  }
  if (!started_) return 0;

  void* ret_val = 0;

  ThreadMainLoopData* tmld = 0;
  ThreadMainLoopData** vvvp= &tmld;
  void** vvv = (void**)vvvp;
  int ret = pthread_join(tid_, vvv);
  if (ret) throw runtime_error("pthread_join");
  if (tmld) {
    ret_val = tmld->return_value;
  }

  delete tmld;
  tmld = 0;

  started_ = false;

  return ret_val;
}

OC_END_NAMESPACE

#define OCTHREAD_H_
#endif //  OCTHREAD_H_
