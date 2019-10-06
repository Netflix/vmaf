
#include "midastalker.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Simple test to try out WaitForReadyMidasTalker.  You need to run a
// MidasServer (midasserver_ex.py for example) on "soc-dualquad1"
//(machine 1) and "brhrts" (machine 2) for this to work.


int main ()
{
  Arr host_port_list = "[ ['soc-dualquad1', 8888], ['brhrts', 8888]]";
  
  // Create a bunch of talkers, talking to different servers
  Array<MidasTalker*> l;
  for (size_t ii=0; ii<host_port_list.length(); ii++) {
    Arr hp = host_port_list[ii];
    string host = hp[0];
    int port    = hp[1];
    MidasTalker* m = new MidasTalker(host, port);
    m->open();
    l.append(m);
  }
  cout << "All listeners ... " << l << endl;
  usleep(1000000); // Give people a chance to wake up and communicate

  // Return immediately all ready: If none are ready, go through one
  // by one and wait
  Array<MidasTalker*> ready = WaitForReadyMidasTalker(l);
  cout << "These listeners have something for me! " << ready << endl;
  
  cout << "Taking data off first ready" << ready[0]->recv() << endl;
  cout << "Taking data off second ready" << ready[1]->recv() << endl;
  
  // 10 second timout, return immediately as soon as one is ready
  ready = WaitForReadyMidasTalker(l, true, 10);
  cout << ready << endl;

  // Cleanup
  for (size_t ii=0; ii<l.length(); ii++) {
    delete l[ii];
  }
  
  return 0;
}
