#ifndef SYNCHRONIZER_H_

// ///////////////////////////////////////////// Synchronizer

extern "C" {
#undef __PURE_CNAME  // some craziness for Tru64, doesn't hurt anyone
#include <errno.h>
}


#include "ocport.h"
#include <stdlib.h>
#include <stdio.h>
#if defined(_MSC_VER)
#define HAVE_STRUCT_TIMESPEC 0
#include <pthread.h>
#include <winsock2.h>
#include <windows.h>
#else
#include <pthread.h>
#include <sys/time.h>
#endif

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

OC_BEGIN_NAMESPACE 


class CondVar; // Forward

// Wrapper for POSIX Mutex
struct Mutex {
  friend class CondVar;
  // Initialize a Mutex
  Mutex (bool shared_across_processes=false) :
    sharedAcrossProcesses_(shared_across_processes) 
  {
    // Initialize Mutex: Because the mutex is probably allocated in
    // shared memory, explicitly inititialize.
    {
      pthread_mutexattr_t mutex_attr;
      if (pthread_mutexattr_init(&mutex_attr)) {
	perror("pthread_mutexattr_init");
	exit(1);
      }
	
      // Necessary so mutexes can be shared between processes.
      if (shared_across_processes) {
#ifndef _POSIX_THREAD_PROCESS_SHARED

# ifndef PTHREAD_PROCESS_SHARED
        // Can't compile this because mutexes/cvs can't be shared 
	errout_("Mutexes can't be shared across processes");
# endif 
	
#endif
	if (pthread_mutexattr_setpshared(&mutex_attr, 
					 PTHREAD_PROCESS_SHARED)) {
	  perror("pthread_mutexattr_setpshared");
	  exit(1);
	}
      }
      // Initialize mutex with these attributes
      if (pthread_mutex_init(&lock_, &mutex_attr)) {
	perror("pthread_mutex_init");
	exit(1);
      }
      // Clean up attribute object
      if (pthread_mutexattr_destroy(&mutex_attr)) {
	perror("pthread_mutexattr_destroy");
	exit(1);
      }
    }
  }
  
  void lock ()        { if (pthread_mutex_lock(&lock_)) { perror("pthread_mutex_lock"); exit(1);} }
  void unlock ()      { if (pthread_mutex_unlock(&lock_)) { perror("pthread_mutex_unlock"); exit(1); } }

  ~Mutex ()
  {
    // Clean up mutex
    if (pthread_mutex_destroy(&lock_)) {
      perror("pthread_mutex_destroy");
      exit(1);
    }
  }

  pthread_mutex_t lock_;
  bool sharedAcrossProcesses_;
}; // Mutex


// This is a helper class to lock and unlock Mutexes automatically so
// thrown exceptions will automatically destruct the ProtectScope and
// unlock the Mutex.
struct ProtectScope {
  ProtectScope (Mutex& m) : mp_(m) { mp_.lock(); }
  ~ProtectScope () { mp_.unlock(); }
  Mutex& mp_;
}; // ProtectScope


// A wrapper for POSIX CondVars.  Note that this is an inplace class:
// none of the data members allocate memory on the heap.
struct CondVar {

    // Create a CondVar with its own Mutex: for simple conditions,
    // this is frequently the easiest way to create a CondVar.  If you
    // plan to use this CondVar across multiple processes in some
    // shared memory, set the "shared_across_processes" to true:  This
    // forces the Mutex to be created for shared memory access.
    CondVar (bool shared_across_processes=false) 
    { init(shared_across_processes); }

    // Sometimes it makes sense to have multiple CondVars associated
    // with a shared mutex (e.g., one CondVar for "queue empty",
    // another for "queue full"): this constructor allows you to share
    // a Mutex that's already been constructed.  If the Mutex was
    // "shared_across_processes", the CondVar will be created the same
    // way.
    CondVar (Mutex& shared_mutex_across_multiple_condvars)
    { 
      lockptr_ = &shared_mutex_across_multiple_condvars.lock_;
      initCondVar_(shared_mutex_across_multiple_condvars.sharedAcrossProcesses_);
    } 
    
    ~CondVar ()
    {
      // Make sure everyone out.  TODO:  make this more robust.
      lock();
      condition_ = true;
      broadcast();
      unlock();

      // Clean up CondVar
      if (pthread_cond_destroy(&cv_)) {
	perror("pthread_cond_destroy");
	exit(1);
      }
      // Clean up mutex (if we own)
      if (lockptr_==&lock_ && pthread_mutex_destroy(&lock_)) {
	perror("pthread_mutex_destroy");
	exit(1);
      }
    }
    
    
    // Initialize the locks and condvars.
    void init (bool shared_across_processes=false)
    {
      initMutex_(shared_across_processes);
      initCondVar_(shared_across_processes);
    }

  
    // Helper routine to initialize ONLY the Mutex
    void initMutex_ (bool shared_across_processes) 
    {   
      lockptr_ = &lock_;
      new (lockptr_) Mutex(shared_across_processes); // inplace construct
    }

    // Helper routine to initialize ONLY the CondVar part
    void initCondVar_ (bool shared_across_processes) 
    {   
      // CVs: Becase the CV may be allocated in shared memory,
      // explicitly initialize
      pthread_condattr_t cv_attr;
      if (pthread_condattr_init(&cv_attr)) {
	perror("pthread_condattr_init");
	exit(1);
      }
      if (shared_across_processes) {
	// Necessary so cvs can be shared between processes.  
	if (pthread_condattr_setpshared(&cv_attr, PTHREAD_PROCESS_SHARED)) {
	  perror("pthread_condattr_setpshared");
	  exit(1);
	}
      }
      // Initialize cv with these attributes
      if (pthread_cond_init(&cv_, &cv_attr)) {
	perror("pthread_mutex_init");
	exit(1);
      }
      // Clean up attribute object
      if (pthread_condattr_destroy(&cv_attr)) {
	perror("pthread_condattr_destroy");
	exit(1);
      }
      condition_ = false;
    }


    void lock ()        { if (pthread_mutex_lock(lockptr_)) { perror("condvar pthread_mutex_lock"); exit(1);}  }
    void unlock ()      { if (pthread_mutex_unlock(lockptr_)) { perror("condvar pthread_mutex_unlock"); exit(1);} }
    void signal ()      { if (pthread_cond_signal(&cv_)) { perror("condvar pthread_cond_signal"); exit(1);} }
    void broadcast ()   { if (pthread_cond_broadcast(&cv_)) { perror("condvar pthread_cond_broadcast"); exit(1); } }
    void wait ()        { if (pthread_cond_wait(&cv_, lockptr_)) { perror("condvar pthread_cond_wait"); exit(1);} }


    // Returns true if a timeout occurred, false if a normal wakeup.
#define NANOSECONDS_IN_SECOND int_u8(1000000000)
#define MICROSECONDS_IN_SECOND int_u8(1000000)
    bool timedwait (int_u8 microsecond_timeout = 1000) // .001 of a second by default 
    {
      // Convert to nano seconds for timespec struct
      int_u8 seconds = microsecond_timeout / MICROSECONDS_IN_SECOND;
      int_u8 microseconds_left = microsecond_timeout % MICROSECONDS_IN_SECOND;
      int_u8 nanoseconds_left = microseconds_left * 1000;

      // Time is an absolute time: the timeout happens AFTER this time.
      struct timespec current_time;


#if defined(_MSC_VER)
        __int64 wintime; GetSystemTimeAsFileTime((FILETIME*)&wintime);
        wintime      -=116444736000000000i64;  //1jan1601 to 1jan1970
        current_time.tv_sec  =wintime / 10000000i64;           //seconds
        current_time.tv_nsec =wintime % 10000000i64 *100;      //nano-seconds
#elif defined(__MACH__) // OS X does not have clock_gettime, use clock_get_time
      clock_serv_t cclock;
      mach_timespec_t mts;
      host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
      clock_get_time(cclock, &mts);
      mach_port_deallocate(mach_task_self(), cclock);
      current_time.tv_sec = mts.tv_sec;
      current_time.tv_nsec = mts.tv_nsec;
#else
      clock_gettime(CLOCK_REALTIME, &current_time);
#endif


      struct timespec timeout = current_time;

      // Add timeout to current time: Handle Rollover of nanoseconds
      timeout.tv_sec += seconds;
      timeout.tv_nsec += nanoseconds_left;
      if (timeout.tv_nsec + nanoseconds_left > NANOSECONDS_IN_SECOND) {
	timeout.tv_sec += 1;
	timeout.tv_nsec = timeout.tv_nsec % NANOSECONDS_IN_SECOND;
      }

      int retcode = pthread_cond_timedwait(&cv_, lockptr_, &timeout);
      return (retcode == ETIMEDOUT );
    }

    bool timedwait_sec (real_8 timeout_in_seconds)
    {
      return timedwait(int_u8(timeout_in_seconds*MICROSECONDS_IN_SECOND));
    }

    volatile bool& operator() () { return condition_; }

    pthread_mutex_t* lockptr_;  // Usually points directly at lock_
    pthread_mutex_t lock_;      // May or may not be used (if sharing mutex)
    pthread_cond_t  cv_;
    volatile bool   condition_;

}; // CondVar 

OC_END_NAMESPACE


#define SYNCHRONIZER_H_
#endif // SYNCHRONIZER_H_
