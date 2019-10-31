#ifndef OCCIRCULARBUFFER_H_

// A CircularBuffer used to hold elements by value: the end the front
// can inserted/deleted in constant time: it can request infinite
// Circular buffer (meaning that puts will never over-write read
// data).

#include "ocport.h"
#include "ocarray.h"

OC_BEGIN_NAMESPACE

template <class T>
class CircularBuffer {
 public:

  // Construct a circular buffer (with buffer length)
  CircularBuffer (int initial_length=4, bool infinite=false, Allocator* a=0) :
    buff_(initial_length, a),
    nextPut_(0),
    nextGet_(0),
    empty_(true),
    infinite_(infinite)
  {
    // Default construct into
    const int capac = buff_.capacity();
    for (int ii=0; ii<capac; ii++) 
      buff_.append(T());
  }

  // Default destructor, copy constructor, op= all good.

  bool empty () const    { return empty_; } 
  bool full () const     { return !empty_ && nextGet_==nextPut_; }
  bool infinite () const { return infinite_; }
  int capacity () const  { return buff_.capacity(); }
  int length () const   
  { 
    if (empty()) { return 0; }
    else if (full()) { return capacity(); }
    else if (nextGet_>nextPut_) { return capacity()-(nextGet_-nextPut_); }
    else { return nextPut_-nextGet_; }
  }

  // Put a single element into the buffer.  If in infinite mode, a put
  // into a "full" buffer will cause it to re-expand and double the
  // size.  If in finite mode, it will throw a runtime_error.
  T& put (const T& c) 
  {     
    checkFullness_();
    // Space available, just plop it in
    T& retval = buff_[nextPut_] = c;
    nextPut_ = (nextPut_+1) % capacity();
    empty_ = false;
    return retval;
  }

  // Get a single element out of the circular buffer.  If the buffer
  // is empty, throw a runtime_error
  T get () 
  {
    if (empty()) { // Empty, can't get anything
      throw runtime_error("Circular Buffer Empty");
    } else {       // nextGet always tells us where we are

      // Get a copy & destroy the old one (so doesn't sit and reclaim memory)
      T& cref = buff_[nextGet_];  // get ref out
      T copy = cref;              // copy
      cref = T();                 // Clean up old value with default construct

      nextGet_ = (nextGet_+1) % capacity();
      empty_ = (nextGet_ == nextPut_);
      return copy;
    }
  }

  // Peek at the nth element (element 0 would be the first thing "get"
  // would return, element 1 would be the next).  Throws the
  // runtime_error exception if try to peek beyond what the buffer
  // has.  This does NOT advance the circular buffer: it only looks
  // inside.
  T peek (int where=0)
  {
    if (where<0 || where>=length()) {
      throw runtime_error("Trying to peek beyond the end of the Circ. Buff");
    }
    int index = (nextGet_+where) % capacity();
    return buff_[index];
  }

  // This implements performing "n" gets, but in O(1) time.  If asked
  // to consume more elements that the CircularBuffer contains, a
  // runtime_error will be thrown.
  void consume (int n)
  {
    if (n<0 || n>length()) {
      throw runtime_error("Trying to consume more data than in Circ. Buff");
    }
    empty_ = (n==length());
    nextGet_ = (nextGet_+n) % capacity();
  }

  // The "get()" always pulls from one side of the circular buffer:
  // Sometimes, you want to be able to pushback some entry
  // you just got as if it were never "get()" ed.   This is
  // very similar to "put", but it is simply doing it on the other
  // side of the circular buffer.  The pushback can fail if the
  // queue is full (not infinite mode) with a runtime_error.
  // If it is an infiite queue, it will simply re-expand.
  T& pushback (const T& pushback_val) 
  {
    checkFullness_();
    // Space available, just plop it in
    nextGet_ = (nextGet_+capacity()-1) % capacity();
    T& retval = buff_[nextGet_] = pushback_val;
    empty_ = false;
    return retval;
  }

  // Drop the last "put()" as if it never went in: this can throw
  // an exception if the buffer is empty.  This will ONLY remain
  // a valid reference until the next non-const operation on the Circ. Q
  T& drop ()
  {
    if (empty()) { // Empty, can't get anything
      throw runtime_error("Circular Buffer Empty");
    } else {       // nextPut always tells us where we are

      // TODO: Do we need to worry about cleaning this up?
      // The old value is still there, and make be taking up
      // precious resources.  Since this interface returns
      // a reference to it, it's up to the user to make
      // sure he cleans is up.  (See get(): we force a T()
      // in place when a get occurs).
      nextPut_ = (nextPut_+capacity()-1) % capacity();
      T& c = buff_[nextPut_];
      empty_ = (nextGet_ == nextPut_);
      return c;
    }
  }

  // Swap with another Array: This is a O(1) operation since we only
  // swap pointers (no copying of full arrays).
  void swap (CircularBuffer<T>& rhs) 
  { 
    buff_.swap(rhs.buff_);
    OC_NAMESPACED::swap(nextPut_,  rhs.nextPut_);
    OC_NAMESPACED::swap(nextGet_,  rhs.nextGet_);
    OC_NAMESPACED::swap(empty_,    rhs.empty_);
    OC_NAMESPACED::swap(infinite_, rhs.infinite_);
  }
    
  // Output
  ostream& print (ostream& os) 
  {
    for (int ii=0, jj=nextGet_; ii<length(); ii++, jj=(jj+1)%capacity()) {
      os << buff_[jj] << " ";
    }
    return os << endl;
  }

 protected:
  Array<T> buff_;
  int nextPut_;          // Points where next put will occur
  int nextGet_;          // Points to where next get will occur
  bool empty_;           // nextPut_==nextGet is either empty or full
  bool infinite_;        // Puts into empty cause a doubling

  // Centralize fullness check and re-expansion code
  void checkFullness_ ()
  {
    if (full()) { // Circ Buffer Full, expand and remake
      if (!infinite()) { 
	throw runtime_error("Circular Buffer full");
      } else {
	// Create a new Circ. Buffer of twice the size
	const int len = buff_.capacity();
	Array<T> temp(len*2);

	// Recopy to the new, larger circ. buffer
	for (int ii=nextGet_, jj=0; jj<len; ii=(ii+1)%len, jj++) {
	  temp.append(buff_[ii]);
	}
	for (int ii=0; ii<len; ii++) // Since twice as big, fill it out
	  temp.append(T());

	// Install new buffer
	buff_.swap(temp);
	nextPut_ = len;
	nextGet_ = 0;
      }
    } // ... and return now that space is available
  }
  
}; // CircularBuffer

// Output
template <class T>
inline ostream& operator<< (ostream& os, CircularBuffer<T>& a)
{ return a.print(os); }

template <class T>
inline void swap (CircularBuffer<T>& lhs, CircularBuffer<T>& rhs)
{ lhs.swap(rhs); }

OC_END_NAMESPACE

#define OCCIRCULARBUFFER_H_
#endif // OCCIRCULARBUFFER_H_

