
#include "shmboot.h"

// Example demonstrating how to use the SHMMain, ServerSide (and
// ClientSide in the clientside.cc file) to have a point to point
// communication from process to process with shared memory.  Vals are
// allocated in shared memory and "sent" from one process to the
// other: because the "pointers" are copied and not a full table, this
// is a very fast way to do interprocess communication.

int main (int argc, char** argv)

{  
  // Probe how memory is used
  void* mem1;
  cerr << "32-bit or 64-bit? " << (sizeof(void*)*8) << endl; 
  cerr << "stack area: " << & mem1 << endl;
  cerr << "heap area : " << ((void*)new char[1]) << endl;
  cerr << "text area : " << (void*)&main << endl;
  FILE* (*fptr)(const char*, const char*) = fopen;
  cerr << "mmap area : " << (void*)fptr << endl;
  cerr << "big heap  : " << ((void*)new char[1000000]) << endl;
  int m;
  cerr << "stack grows " <<( ((((char*)&m) - ((char*)mem1)) < 0) ? "high to low" : "low to high") << endl; 

  // On Linux machines, stack typically grows from high to low (i.e., down),
  // and usually right next to kernel barrier (0x800000000000 on 64-bit
  // machine, and 0xC0000000 on 32-bit machines).  Everything else
  // (text, BSS, heap, mmap) tends to be at lower addresses.
  // Yes, this is non-portable, but placing shared memory 
  // in a location where no-one else uses it and that is 
  // the same across non-inheritance related process is difficult!
  if (sizeof(void*)==4) { // 32-bit
    // Kernel starts at 0xC0000000, grows up from there: stack
    // Stack starts at 0xBfffxxxx (maybe guard page between kernel and stack)
    // Gives 268 Meg for shared memory and stack to share ...
    mem1 = (void*)0xB0000000; 
  } else if (sizeof(void*)==8) {// 64-bit 
    // ... (actually only 48-bits of address space in current gen of Intel)
    // Kernel starts at 0x800000000000, grows low to high 
    // Stack starts at  0x7fffffffxxxx, grows high to low
    // Gives 104Tbytes for shared memory and stack to share...
    mem1 = (void*)0x700000000000ULL;  
  } else { // 
    cerr << "Unknown model" << endl;
    exit(1);
  }


  cout << "******** FOR REPRODUCIBLE RESULTS: May have to start this as\n";
  if (sizeof(void*)==4) {
    cout << "  % setarch i386 -L -R " << argv[0] << " ...\n";
  } else if (sizeof(void*)==8) {
    cout << "  % setarch x86_64 -L -R " << argv[0] << " ...\n";
  } else {
    cout << " ???  Unknown arch ??? use uname -i to find your arch " 
	 << " and use " << endl
	 << "  % setarch xxx -L -R " << endl
	 << "     where xxx is the result of 'uname -i'" << endl;
  }
  cout << "***********************************************" << endl;

  if (argc!=5) {
    cerr << "Usage:" << argv[0] 
	 << " shared_memory_name bytes pipename pipe_capacity" << endl
	 << "      -shared_memory_name is string name of shared memory pool\n"
	 << "      -bytes to allocate in shared memory region\n" 
	 << "      -pipename is the string name clientside uses to find pipe\n"
	 << "      -capacity is number of packets pipe can enqueue until blocks"
	 << endl;
    exit(1);
  }
  string shared_mem_name = argv[1];
  size_t bytes = atoi(argv[2]);
  string pipename = argv[3];
  size_t capacity = atoi(argv[4]);
  bool debug = true;
  
  // (1) Some process (not necessarily this one, but some 
  //     process that will be alive for entire application)
  //     creates the shared memory.
  SHMMain mem(shared_mem_name, bytes, debug, mem1);
  mem.start();  // Have to start to create regions

  // (2) A Serverside talks point to point with a client
  ServerSide server(shared_mem_name, pipename, capacity, debug);
  server.start();
  CQ& pipe = server.pipe(); // Once started, can get communication point

  // (3) ... And start talking!
  for (int ii=0; ii<100; ii++) {
    
    // Create some table in shared memory
    Val data = Shared(server.pool(), Tab());
    data["sequence_number"] = ii; // ... anything inserted into this 
                                  // table should end up in shared memory
    data["bi"] = int_n("100000000000000000000000000000000000000000000000");
    data["bu"] = int_un("100000000000000000000000000000000000000000000000");
    
    // Use timeout: that way you won't get stuck in a condvar that
    // never returns.
    bool enqueued = false;
    while (!enqueued) {
      enqueued = pipe.enqueue(data, .2);
    }
    
    // Enqueued, go to next data
  }
  
}
