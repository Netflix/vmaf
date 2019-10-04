#ifndef OCCQTS_H_

// Includes

#include "ocval.h"
#include "octhread.h"
#include "occircularbuffer.h"

OC_BEGIN_NAMESPACE

// A templated circular Queue with synchronization.
// All data is held by value (like CircularBuffer).
template <class T>
class CQThreadSafe {
 public:

  // Create a circular queue of the appropriate length
  CQThreadSafe (int len, Allocator *a=0, bool shared_across_process=false) :
    q_(len, false, a),
    m_(shared_across_process),    // Initialize mutex:  ORDER DEPENDENCY
    full_(m_),  // --Have condvars full and empty share mutex,
    empty_(m_)  // --both associated with same q
  { }

  // Enqueue the packet, blocks if not enough room
  void enq (const T& data)
  {
    // Can't enque if nothing there ... block
    m_.lock();
    while (q_.full()) {
      full_.wait();
    }
    q_.put(data);
    empty_.broadcast();
    m_.unlock();
  }

  // Returns true if it enqueued within the time limit,
  // otherwise, returns false to show it didn't queue.
  // None means blocking (i.e., no timeout).
  bool enqueue (const T& data, Val timeout_in_seconds=None)
  {
    // None means no timeout, so it blocks forever
    if (timeout_in_seconds == None) {
      this->enq(data);
      return true;
    }

    // Can't enque if nothing there ... block
    m_.lock();
    while (q_.full()) {
      bool timeout = full_.timedwait_sec(timeout_in_seconds);
      if (timeout) {
	m_.unlock();
	return false;
      }
    }
    q_.put(data);
    empty_.broadcast();
    m_.unlock();
    return true;
  }

  // Dequeue, blocks if no room
  T deq ()
  {
    // Can't deque if nothing there ... block
    m_.lock();
    while (q_.empty()) {
      empty_.wait();
    }
    T data = q_.get();
    full_.broadcast();
    m_.unlock();
    
    return data;
  }

  // Dequeue: if timeouts, returns false (indicating value
  // is invalid).  If true, got value value in timely fashion.
  bool dequeue (Val timeout_in_seconds, 
		T& dequeued_value)
  {
    if (timeout_in_seconds==None) {
      dequeued_value = this->deq();
      return true;
    }

    // Can't deque if nothing there ... block
    m_.lock();
    while (q_.empty()) {
      bool timed_out = empty_.timedwait_sec(timeout_in_seconds);
      if (timed_out) {
	m_.unlock();
	return false;
      }
    }
    dequeued_value = q_.get();
    full_.broadcast();
    m_.unlock();

    return true;
  }

  bool empty ()
  {
    bool retval;
    m_.lock();
    retval = q_.empty();
    m_.unlock();
    return retval;
  }

  // protected:
  CircularBuffer<T> q_;
  Mutex m_;        // Protects q:  used with both full_ and empty_ below
  CondVar full_;   // Mark when queue is full, so have to wait to put
  CondVar empty_;  // Mark when queue is empty, so have to wait to get

}; // CQThreadSafe

OC_END_NAMESPACE

#define OCCQTS_H_
#endif // OCCQTS_H_
