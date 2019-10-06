
#include "ocproxy.h"
#include "ocval.h"
#include "ocstreamingpool.h"
#include "ocpermutations.h"
#include "octhread.h"
#include "sharedmem.h"

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


#define BYTES_FOR_INIT 4
#define MAX_ENTRIES 1000

struct Bootstrap_t {
  // The main memory segment must be mapped to the same address in
  // every process: this records where it is created.
  StreamingPool* main_pool;
  
  // We'll allocate a table immediately from main segment and put it
  // here so both have access to it.
  Proxy p;

  // Tell use when to start
  CondVar cv;

  // extra bytes for initialization  (only use last)
  char initializtion[BYTES_FOR_INIT];
}; // Bootstrap_t


// Some global names
string bootstrap = "/bootstrap"; // "/usr/tmp/bootstrap";
size_t bytes_in_boot = sizeof(Bootstrap_t);

string name = "/my_mem"; // "/usr/tmp/my_mem";
size_t bytes_in_shm = 1024*1024+BYTES_FOR_INIT;
size_t bytes_in_pool = 1024*1024;

// Process of parent
void Parent (bool joining=true)
{
  StreamingPool* shm = 0;
  {
    // create a small bootstrap memory pool:  In this pool, we will
    // have the starting conditions
    Bootstrap_t* boot = 
      (Bootstrap_t*)SHMCreate(bootstrap.c_str(), bytes_in_boot);
    if (boot==0) {
      cerr << "ERROR: couldn't create boot pool" << endl;
      exit(1);
    } else {
      cerr << "...boot pool created" << endl;
    }
    

    // Create the pool  
    char* mem = (char*)SHMCreate(name.c_str(), bytes_in_shm);
    if (mem==0) {
      cerr << "ERROR: couldn't create main pool" << endl;
      exit(1);
    } else {
      cerr << "...main pool created" << endl;
    }
    shm=StreamingPool::CreateStreamingPool(mem, bytes_in_pool);
    
    // Initialize so other pool can see
    boot->main_pool = shm;
    Proxy p = Shared(shm, Tab("{0:1.1, 1:2.2, 'hello':'there!'}"));
    new (&boot->p) Proxy(p);
    new (&boot->cv) CondVar(true);
    
    // Tell other pool ready to go
    SHMInitialize(boot, bytes_in_boot);
    SHMInitialize(mem, bytes_in_shm);
    
    // Keep doing stuff
    while (1) {
      TransactionLock tl(p);
      
      Tab& t = p;
      int entries = t.entries();
      if (entries>MAX_ENTRIES) break;
      if (entries % 2 == 0) {
	t.append("parent");
	//cerr << "parent:" << t << endl;
      } else {
	continue;
      }
    }
    
    // Wait for child to return
    int status = 0;
    struct rusage ru;
    if (joining) {
      wait3(&status, 0, &ru);
    }

    cerr << ((joining)? "JOINED" : "RETURNING") << " with status " << status << " ... now cleaning up!" << endl;
    
    boot->~Bootstrap_t();
  }
  cerr << "Memory clean?" << shm->isPristine() << endl;

  SHMUnlink(bootstrap.c_str());
  SHMUnlink(name.c_str());
}


void Child (bool joining=true)
{
  if (joining) { } ; // dumb test to make sure you use var
  StreamingPool* child_shm = 0;
  {
    // Try to map in the bootstrap
    size_t bbytes;
    Bootstrap_t* boot = 
      (Bootstrap_t*)SHMAttach(bootstrap.c_str(), NULL, bbytes);
    if (boot==0) {
      cerr << "Um, couldn't attach to the bootstrap session" << endl;
      exit(1);
    }
    if (bytes_in_boot!=bbytes) {
      cerr << "Error: Expected the number of bytes for bootstrap pool to be the same" << endl;
    }
    
    // Child waits for boot to be ready
    while (!SHMInitialized(boot, bytes_in_boot)) {
      // make sure memory maps before we start looking at it
      usleep(10000);  // wait 1/10th of second 
    }
    cerr << "...child connected to bootstrap mem" << endl;

    // Assertion: boot area mapped in and initialized
    void* start = boot->main_pool;
    
    // Child needs to attach to pool
    size_t bytes;
    void* mem = SHMAttach(name.c_str(), start, bytes);
    if (mem==0) {
      cerr << "um, couldn't attach to the main session" << endl;
      exit(1);
    }
    if (bytes!=bytes_in_shm) {
      cerr << "Error: Expected the number of bytes for bootstrap pool to be the same" << endl;
    }
    child_shm = (StreamingPool*) mem;
    
    
    // Child waits for boot to be ready
    while (!SHMInitialized(mem, bytes_in_shm)) {
      // make sure memory maps before we start looking at it
      usleep(10000);  // wait 1/10th of second 
    }
    cerr << "...child connected to main mem" << endl;
    
    Proxy p = boot->p;
    cerr << "The table:" << p << endl;
    
    while (1) {
      {
	TransactionLock tl(p);
	
	Tab& t = p;
	int entries = t.entries();
	if (entries>MAX_ENTRIES) break;
	if (entries %2 == 1) {
	  t.append("child");
	  // cerr << "Child:" << t << endl;
	} else {
	  continue;
	}
      }
    }
    
    cerr << "Child exiting" << p << endl;
  }
  cerr << "Memory pristine?" << child_shm->isPristine() << endl;
}

void ProcessTest (int how)
{
  if (how==-1) {
    // SPLIT!!
    pid_t id = fork();
    if (id==-1) { throw runtime_error("FAILED!"); }
    
    // parent
    if (id) {    
      Parent();
    } else {
      Child();
    }
  }

  else if (how==0) {
    Parent(false);
  } else {
    Child(false);
  }
}

int main (int argc, char* argv[])
{
  if (argc!=2) {
    cerr << "usage:" << argv[0] << " [-1|0|1]" << endl;
    cerr << "        where -1 means fork:parent and child run from here\n";
    cerr << "        where 0  means runs parent only\n";
    cerr << "        where 1  means run child only" << endl;
    string m = 
      " IMPORTANT NOTES!  \n"
      " (1) Make sure the shared memory region on disk is cleaned up:\n" 
      "     If everything works correctly, the server creates both \n"
      "       /dev/shm/bootstrap and /dev/shm/my_mem   (under Linux)\n"
      "     and the client connects to them both, then both files should\n"
      "     be removed at the end of the execution.  If something weird\n"
      "     happens, you may need to clean those files up:\n"
      "   % rm  /dev/shm/bootstrap /dev/shm/my_mem\n"
      "\n"
      " (1) If you run this test as a separate client and server process \n"
      "     (i.e., no forking), be careful.  On a kernel that supports the \n"
      "     'RedHat Address Randomization', you must turn that feature off\n" 
      "      or this example won't probably work.  You can do that with \n"
      "   % setarch i386 sharedmem_test 0  # On the server process when run\n"       "   % setarch i386 sharedmem_test 1  # On the client process when run\n" ;
    cerr << m << endl;
    exit(1);
  }
  int opt = atoi(argv[1]);
  ProcessTest(opt);
}
