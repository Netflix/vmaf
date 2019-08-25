#ifndef OCBAG_H_

// An implementation of a simple thread-neutral/thread-safe bags where data
// can be pulled from the bag in parallel by multiple threads concurrently.  
// There are two basic implementations (note they all start with i to show 
// they only dole out ints):
// The iDrawer: the most basic threadsafe primitive for doling out ints: there
//              is no notion of how many workers may be trying to pull from
//              the drawer.  It has a very simple interface and tends to be
//              thread-neutral except when too many threads pull too quickly.
// The iCupboard: An array of drawers where each worker nominally gets 
//              its own drawer (avoiding interference among drawers).  This 
//              abstraction has a notion of how many workers are pulling data: 
//              usually 1 per drawer.  The interface is slightly more clumsy.
//
// Use whatever makes the most sense for your application.
//
// NOTE: You'll note that both abstractions *ONLY* dole out integers
// from 0..n-1: the key observation is that if you want elements of something
// doled out, simply put them in an array and dole out based on int.
// For example:
//
//  Arr a = "[ 'work1', 'larger work2', 'really really large work3']";
//  iBag bag(a.length());  // Bag of 3 integers: { 0, 1, 2 } 
//  int_u4 item = -1;
//  while (bag.get(item)) { 
//       Val& single_piece_of_work = a[item];
//       doSomething(single_piece_of_work);
//  }
//
// Take a look at bag_test.cc in the tests area for usage and examples.
// 
// Why are there two abstractions?  The first is the iDrawer: it is the
// fundamental threadsafe primitive that the other is built from.  Most 
// of the time, the iDrawer will probably be good enough and the interface is 
// simple, but if you are doing lots of quick pulls from the bag, the iCupboard
// may be the choice; it tends to keep each worker out of the way of the 
// others (although the iCupboards are slightly harder to use). 
// Summary:  
//  * iDrawer: SIMPLE interface, small # workers, lots of work per pull
//  * iCupboard: more complex interface, any number workers, less work per pull
// Basically, start with iDrawer and see how well it scales for you.
// Google for the paper "Thread-safe and Thread-Neutral Bags" for more
// discussion.

#include "ocsynchronizer.h"
#include "ocarray.h"

OC_BEGIN_NAMESPACE

// Consider using FASTREFCOUNT if you are on a later GCC: it can speed
// up the performance of the Cupboard by 2x in some cases.
#define FASTREFCNT

// A simple threadsafe bag with no notion of how many workers may be accessing 
// it.   It doles out integers from [seq_start, seq_start+seq_length)
// This is a primitive to build Cupboards and Bags from.
class iDrawer {
  
 public:
  // Initialize a drawer
  iDrawer (int seq_start, int seq_length, bool protect=true) :
    current_(seq_start),
    upperBound_(seq_start+seq_length),
    protect_(protect)
  { }

  // Get an item out of this drawer: returns true if something
  // was got (assigned into item) or false if the drawer was empty
  bool get (int_u4& item)
  {
    if (current_ >= upperBound_) {
      return false;
    }
    if (protect_) {
#ifndef FASTREFCNT 
      {
	ProtectScope ps(lock_);
	item = current_++;
	if (item >= upperBound_) {
	  return false;
	}
      }
#else 
      // item = current_;
      item = __sync_fetch_and_add(&current_, 1);
      if (item >= upperBound_) {
	return false;
      }
#endif
    } else {
      item = current_++;
    }
    return true;
  }

  // See if empty
  bool empty () { return current_ >= upperBound_; }

 protected:
  
#ifndef FASTREFCNT
  Mutex lock_;         // Lock for the drawer (protects the array contents)
#endif

  // Contents of drawer
  volatile int_u4 current_;      // frequently looked at

  char buf1[64];                 // upperBound is looked at enough
                                 // eliminate false sharing between
                                 // current and uppberBound: processors
                                 // can cache upperBound

  int_u4 upperBound_;            // set once at construction, 
                                 // only read from then on

  // DO NOT use any atomic incrementing: a plain cupboard with just
  // one worker per drawer (where he NEVER changes drawers) is a 
  // simple barrier sync problem and avoids synchronization primitives.
  bool protect_;

  char buff[64]; // eliminate cache line sharing or false sharing BETWEEN 
                 // drawers

}; // iDrawer


// A thread-hot cupboard containing only ints from 0 .. n-1
class iCupboard {
  
 public:

  // A cupboard containing all integers from 0 .. n-1.  You can
  // specify the number of drawers.  Usually, you want the number
  // of drawers to be the number of threads that you anticipate looking
  // through the drawers.
  iCupboard (int_u4 n, int drawers, bool protect=true) :
    n_(n),
    drawers_(drawers),
    protect_(protect)
  { 
    // Divy up the work so each drawer has "about" the same
    int_u4 work_per_drawer = n / drawers;
    int_u4 leftover_work   = n % drawers;
    
    int_u4 current_work = 0;
    for (int ii=0; ii<drawers; ii++) {
      // Distribute leftover work (at most 1 per drawer)
      int_u4 extra = 0;
      if (leftover_work) {
	extra = 1;
	leftover_work -= 1;
      }
      
      // Get drawer and fill with work_per_drawer
      iDrawer d(current_work, work_per_drawer + extra, protect);
      current_work += work_per_drawer + extra;
      
      // Drawer filled: put in cupboard
      drawers_.append(d);
    }
  }
 

  // Get an item from SOME drawer: start looking from the given drawer. 
  // If we find something, return true with the item and which drawer 
  // we found it in.  Otherwise, return false (meaning cupboard is empty).
  bool get (int starting_drawer, 
	    int_u4& item, int& ending_drawer)
  {
    const int len = drawers_.length();
    int drawer = starting_drawer;
    for (int ii=0; ii<len; ii++) {
      if (drawers_[drawer].get(item)) { // Found a drawer with stuff!
	ending_drawer = drawer;
	return true;
      }
      if (!protect_) break;
      drawer = (drawer+1) % len;
    }
    return false;
  }

  // Like get, but don't care what drawer we found the item in.
  bool get (int starting_drawer, int_u4& item) 
  {
    int unused;
    return get(starting_drawer, item, unused); 
  }

  // Empty?  Have to look through all drawers.
  bool empty () 
  {
    const int len = drawers_.length();
    for (int ii=0; ii<len; ii++) {
      if (!drawers_[ii].empty()) return false;
    }
    return true;
  }

  // Number of drawers
  int drawers () const { return drawers_.length(); }

  // What integers are we serving out? 0..n-1
  int n () const { return this->n_; }

  // Get a refer to the drawer
  iDrawer& operator[] (int i) { return drawers_[i]; }

 protected:

  // The numbers [0..n) aka 0..n-1 are in the cupboard
  int_u4 n_;

  // A Cupboard is a simple array of drawers, where each drawer
  // will contain some of the supplies.
  Array< iDrawer > drawers_;

  // Only look in one drawer
  bool protect_;

}; // iCupboard


OC_END_NAMESPACE

#define OCBAG_H_
#endif //  OCBAG_H_
