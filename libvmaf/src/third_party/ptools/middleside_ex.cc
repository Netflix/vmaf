
#include "shmboot.h"

// Shows how to write a transformer (something that reads and writes)
// using the ServerSide and ClientSide classes.  See serverside_ex.cc
// for how to write a generator and clientside_ex.cc for how to write
// a Analyzer.

int main (int argc, char** argv)
{
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
	 << " shared_memory_name input_pipename output_pipename outpipe_capacity " << endl
	 << "      -shared_memory_name is string name of shared memory pool\n"
	 << "      -input_pipename is the pipe we are reading from\n"
	 << "      -output_pipename is the pipe we are writing to\n"
	 << "      -capacity is number of packets outpipe can enqueue until blocks"
	 << endl;
    exit(1);
  }
  string shared_mem_name = argv[1];
  string input_pipename = argv[2];
  string output_pipename = argv[3];
  size_t out_capacity = atoi(argv[4]);
  bool debug = true;
  
  // (0) ... some other process created SHMMain ... 

  // (1) Client which reads from some other process
  ClientSide client(shared_mem_name, input_pipename, debug);
  client.start();
  CQ& input = client.pipe(); 

  // (2) Server which writesto output pipe
  ServerSide server(shared_mem_name, output_pipename, out_capacity, debug);
  server.start();
  CQ& output = server.pipe(); 


  // (3) ... And start talking!
  while (1) {

    // **** Input
    // Use timeout: that way you won't get stuck in a condvar that
    // never returns.
    Val data;
    bool dequeued = false;
    while (!dequeued) {
      dequeued = input.dequeue(.2, data);
    }
   
    // Dequeued ... do something to data
    data["my_sequence"] = int(data("sequence_number")) + 100;
    data.prettyPrint(cout);
   
 
    // ***** Output
    // Use timeout: that way you won't get stuck in a condvar that
    // never returns.
    bool enqueued = false;
    while (!enqueued) {
      enqueued = output.enqueue(data, .2);
    }
    
    // Enqueued, go to next data
    if (int(data("my_sequence")) == 199) break;
  }
  
}
