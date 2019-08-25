#ifndef SHMBOOT_H_
#define SHMBOOT_H_

// This presents abstractions for making it easier to deal with shared memory
// packets a "queue".  This takes advantage of the fact that Vals can 
// exist in shared memory across processes.  Because this maps shared memory
// across processes in, you have to be careful in the creation of your 
// processes (to avoid the RedHat address randomization feature, you may
// have to start your processes with "setarch -i386 -L -R: see the
// serverside_ex.cc and clientside_ex.cc for examples).

#include "ocproxy.h"
#include "ocval.h"
#include "ocstreamingpool.h"
#include "octhread.h"
#include "sharedmem.h"
#include "occircularbuffer.h"
#include "occq.h" 


#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <stdint.h>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

PTOOLS_BEGIN_NAMESPACE

// Debug routine
void memdump(void* memo, size_t bytes);

#define BYTES_FOR_INIT 4

// Bootstrap region that maps some shared info between server
// and client.  It is a small region that specifies some paramaters
// of larger region.  This overlays a small shared memory region.
// Inside, this specifies how big the BIG memory region has to be.
struct Bootstrap_t {

  // Number of bytes in the MAIN pool: the client sees this when he attaches
  // so the server can control the size of the pool dynamically
  volatile size_t bytes_in_main_pool;

  // The main memory segment must be mapped to the same address in
  // every process: this records where it is created.
  volatile StreamingPool* main_pool;

  // Some data
  volatile void* data;
  
  // extra bytes for initialization  (only use last)
  char initialization[BYTES_FOR_INIT];
}; // Bootstrap_t


// State of the SHM: have we started?  
enum State_e { STOPPED, STARTING, STARTED };
  
// Helper class: both client and server need most of same info
class SHMStub_ {
    
 public:

  // Stub shared by both client and server: both need to know
  // names 
 SHMStub_(const string& name, size_t bytes, bool debug=false, 
	  void *forced_memory_region=0,
	  BreakChecker external_break=0, int_8 micro_sleep=int_8(1e5)) :
    bootname_(name+"_boot"),
    name_(name),
    bytesInMainPool_(bytes),
    debug_(debug),
    forcedAddr_(forced_memory_region),
    state_(STOPPED),
    bootSection_(0),
    mainPool_(0),
    warning_(true),
    externalBreak_(external_break),
    microSleep_(micro_sleep)
  { }

  virtual ~SHMStub_ () { }

  // Return the shared memory pool
  StreamingPool* pool() 
  { 
    if (state_==STARTED && mainPool_) {
      return mainPool_; 
    } else {
      throw runtime_error("Shared memory not started yet");
    }
  }

  // Return the boot
  Bootstrap_t* boot () 
  {
    if (state_==STARTED && bootSection_) {
      return bootSection_;
    } else {
      throw runtime_error("Shared memory not started yet");
    }
  }

  // Return the current state
  State_e state () const { return state_; }


  // Can we detect that the Address Randomization feature is on or off?
  bool addressRandomization () const;

  // Turn on/off address randomization error message.
  // If we *think* the feature is on, then its very unlikely that the
  // SHMMain/ClientSide/ServerSide code will work, so we will issue a 
  // warning.  It's difficult to detect EXACTLY that setarch has been
  // used (its a heuristic which may give a false positive), so we 
  // need to be able to turn off the message.
  bool warning (bool state) { bool old = warning_; warning_=state; return old; }

 protected:

  // ///// Data Members

  string        bootname_;     // Name of (very small) bootstrap memory region
  string        name_;         // Name of the main shared memory region 
  size_t        bytesInMainPool_;  // size of main memory segment
  bool          debug_;        // Are we debugging?
  void*         forcedAddr_;   // 0 if letting system choose, set otherwise
  State_e       state_;        // Current state of the server
  Bootstrap_t   *bootSection_; // Pointer to where boot starts (invalid in client)
  StreamingPool *mainPool_;    // Pointer to where shared mem is 
  bool           warning_;     // If we detect lack of setarch, output warning

  BreakChecker  externalBreak_; // External condition check: if true, abandon ship
  int_8          microSleep_;    // When looping, how long to sleep in micro secs

  // ///// Members

  // Check to see if address randomization is on: if it is,
  // output a big warning about how to turn it off (if the warning flag is set)
  // Note that this is virtual in case you want to change it  ... 
  virtual void checkAddressRandomization_ ();

}; // SHMStub_



// We decouple the shared memory creation from the Server/Client creation.
// Some "global" entity (usually the first component, before the m_sync)
// will create the memory.  From then on, servers and clients use the
// memory.  Servers create the pipe (CQ in shared memory) and clients
// connect to it.

// /// MAIN 
//   SHMMain   memory("shared_memory_pool_name", bytes_in_memory_pool);    
//   memory.start();
//   m_sync();

// /// Servers
//   SHMServer server("shared_memory_pool_name", "pipename", capacity);     
//   m_sync();  // After main server memory is up
//   server.start();
//   while (!Mc->break_) {
//       Val data = ... get data somehow ...
//       bool enqueued=false;
//       while (!enqueued) {
//          enqueued = server.pipe().enqueue(data, 2.3);  // 2.3 seconds timeout
//          if (Mc->break_) exit(1);
//       }
//   }

// /// Clients
//
//    SHMClient client("shared_memory_pool_name", "pipename");
//    m_sync();  // After main server memory is up
//    client.start();
//    while (!Mc->break_) {
// 
//       Val data;
//       bool valid = client.pipe().dequeue(data, 5.0); // 5 second timeout
//       if (valid) { // data dequeued
//          ... do something with valid data ...
//       } else { // nothing, try again
//          continue;
//       }
//    }

// The typical protocol: 
// (1) "Main" server allocates shared memory (usually in first component
//     before the m_sync in X-Midas).
//     a) allocates the Boot pool, mmaps it
//     b) Allocate the main pool and initialize: mmaps it
//     c) fill in boot area to point to this
//
// (2) Each server/client maps in the boot section INTO SOME
//     RANDOM place in memory.  This tells
//     where the main memory must be mapped into the process.
//     Get this information from the boot.  unmap the boot
//     (in case they would intersect).
//


// (3) Each server allocates a CQ which clients can look up
//     to see where it is.  After the server has created
//     the mapping of pipename->CQ, put it in the data structure
//     so clients can look it up.
//

// (4) After CQ is allocated, clients connect.  Until the pipe is
//     created, they keep trying to connect.




// ////////////////////////////////////////// SHMMain

// Manages creation of the boot pool and the main memory pool
class SHMMain : public SHMStub_ {

  public:

  // Set up shared memory region with the given name
  // with the given number of bytes.  Note that the given name
  // usually should be a path like /mymem (it is usually a 
  // memory region under /dev/shm/, so /mymem ends up being 
  // a file /dev/shm/mymmem.
  //
  // The memory isn't actually created until the start
  SHMMain (const string& memory_pool_name, size_t bytes, 
	   bool debug=false, void* forced_addr=0,
	   BreakChecker external_break=0, int_8 micro_sleep=int_8(1e5)) :
    SHMStub_(memory_pool_name, bytes, debug, forced_addr, external_break, micro_sleep)
  { 
    // Make sure the OLD ones are all gone, in case they were left
    // over
    try {
      SHMUnlink(bootname_.c_str(),false);
      SHMUnlink(name_.c_str(),false);
    } catch (...) {
    };
  }

    
  // Be careful when you call the destructor to a predictable time. 
  virtual ~SHMMain ()
  {
    bootSection_->~Bootstrap_t();
    if (debug_) {
      cerr << "Memory clean?" << mainPool_->isPristine() << endl;
    }

    SHMUnlink(bootname_.c_str());
    SHMUnlink(name_.c_str());
  }

  // Start and create the shared memory segment.  
  virtual State_e start ();

 
}; // SHMMain

// ////////////////////////////////////////// SHMConnector_ 

// Attach to the shared memory pools.  This is used by both 
// the Clients and Servers: the Servers have the added responsibility
// of creating a pipe, and the clients have to wait for the server
// to finish creating it.
class SHMConnector_ : public SHMStub_ {

 public:
  
  //  The name given here has to match what the server has set
  SHMConnector_ (const string& memory_pool_name, const string& pipename, 
		 bool debug=false, void* forced_addr=0,
		 BreakChecker external_break=0, int_8 micro_sleep=int_8(1e5)) :
    SHMStub_(memory_pool_name, 0, debug, forced_addr, external_break, micro_sleep),
    pipename_(pipename),
    queue_(0)
  { }

  virtual ~SHMConnector_ () 
  { }

  // Attach to the shared memory.  
  virtual State_e start ();

  // Return the cq, once everything is initialized
  CQ& pipe () 
  {
    if (queue_) {
      return *queue_; 
    } else {
      throw runtime_error("Pipe:" +bootname_+ "/"+name_+" not initialized yet");
    }
  }

 protected:

  string pipename_;            // the name of the pipe
  CQ    *queue_;               // pointer to pipe in shared memory
  Proxy *nameToPipeProxyPtr_;  // pointer to proxy to table

}; // SHMConnector_



// ////////////////////////////////////////// ServerSide

// The Server side of a pipe: this causes the pipe to be created
// in shared memory with the given pipename and the given capacity:
// All the work happens insde the start.
class ServerSide : public SHMConnector_ {
  
 public:
  // Create a shared memory region managed by a CircularQueue with
  // capacity elements
  ServerSide (const string& memory_pool_name, 
	      const string& pipename, int pipe_capacity, 
	      bool debug=false, void* forced_addr=0,
	      BreakChecker external_break=0, int_8 micro_sleep=int_8(1e5)) :
    SHMConnector_(memory_pool_name, pipename, debug, forced_addr, external_break, micro_sleep),
    pipeCapacity_(pipe_capacity) 
  { }

  // Start the server side
  virtual State_e start ();

 protected:
    
  int pipeCapacity_;      // the max capacity (in packets): server sets

}; // ServerSide

// ////////////////////////////////////////// ClientSide

// The ClientSide of a pipe.  This watches the given poolname for the
// appropriate pipename to be created, at which point the client
// is attached.  Note that that this is a point to point attachment:
// if one client removes a packet from the queue, other queues can't
// see it.  All work happens in start.
class ClientSide : public SHMConnector_ {
 public:
  ClientSide (const string& memory_pool_name, const string& pipename, 
	      bool debug=false, void* forced_addr=0,
	      BreakChecker external_break=0, int_8 micro_sleep=int_8(1e5)) :
    SHMConnector_(memory_pool_name, pipename, debug, forced_addr, external_break, micro_sleep)
  { } 

  // Attach and get the pipe being used
  virtual State_e start ();

}; // ClientSide


PTOOLS_END_NAMESPACE

#endif // SHMBOOT_H_
