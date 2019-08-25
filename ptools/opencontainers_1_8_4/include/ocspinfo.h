#ifndef OCSPINFO_H_
#define OCSPINFO_H_


// This is an implementation detail for implementing StreamingPool.
// Users should not have to see this: because of the forward nature of
// a memory allocator, this needs to be in its own file so the entire
// defintion is available.

#include "ocport.h"

OC_BEGIN_NAMESPACE

// Forward
class StreamingPool; 
class SPInfo_;
SPInfo_* GetFreeList (StreamingPool*);  // Forward because of cyclic dependencies

// Sentinel at the front and end of memory to show we can't go any further:
// small int possible at the size of the int_ptr:  This is either
// INT_MIN (32-bit) or LONG_MIN (64 bit).
// This is 2's complement smallest int: 1 bit only in MSB
const int_ptr MEMORY_BOUNDARY=int_ptr(int_ptr(1) << ((sizeof(int_ptr)<<3)-1) );

// The overhead associated with an allocation of memory.  Every piece
// of user memory has a "SPInfo_" overlaid on top.  The user should
// NEVER see this class: it's just an implementation detail.
class SPInfo_ { 

  friend class StreamingPool;

 public: // protected:

  // Get where the USER memory starts in an INUSE node
  char* userData () { return (char*) &prev_; } // TODO: Alignment?

  int_ptr length () const
  {
    int_ptr total_bytes = -totalBytes_;
    if (total_bytes < 0) {
      throw runtime_error("Only expected length on free nodes");
    }
    return total_bytes - 2*sizeof(totalBytes_) - sizeof(pool_);
  }

  // Given a piece of memory as seen by the user, give the front of
  // the struct since the user data is in the middle
  static SPInfo_* GetSPInfo (char* start_of_user_data)
  {
    size_t user_data_offset = sizeof(int_ptr) + sizeof(StreamingPool*);
    char* sp = start_of_user_data - user_data_offset;
    SPInfo_* spinfo = (SPInfo_*)sp;
    return spinfo;
  }


  // Is this node inuse?
  bool isFree () const { return totalBytes_<0; }
  bool isInuse () const { return totalBytes_>0; }

  // Give me the next contiguous node after this one in memory
  SPInfo_* nextContiguous () const 
  {
    char* current = (char*) this;
    current += (totalBytes_<0) ? -totalBytes_ : +totalBytes_;
    SPInfo_* sp = (SPInfo_*) current;
    if (sp->totalBytes_==MEMORY_BOUNDARY) return 0;  // Final 4 bytes has a sentinel
    return sp;
  }

  // Give me the previous contiguous node before this one in memory
  SPInfo_* prevContiguous () const 
  {
    char* current = (char*) this;
    char* c = current - sizeof(totalBytes_);
    int_ptr* total_bytes_of_previous = (int_ptr*) c;
    int_ptr bytes = *total_bytes_of_previous;
    if (bytes==MEMORY_BOUNDARY) return 0; // Sentinel (freelist) marking bound
    bytes = (bytes<0) ? -bytes : bytes;
    current -= bytes;
    SPInfo_* sp = (SPInfo_*) current; 
    return sp;
  }

  // Give me the next node on the freelist
  SPInfo_* nextFree () 
  {
    return next_;
  }


  // Get a reference to the total bytes field at the end
  int_ptr& endTotalBytes () 
  {
    int_ptr total_bytes = totalBytes_;
    if (total_bytes<0) total_bytes = -total_bytes;
    char* current = (char*) this;
    current  = current + total_bytes - sizeof(totalBytes_);
    int_ptr* end = (int_ptr*)current;
    return *end;
  }

  // Set the total number of bytes, on both front end and back end
  void setTotalBytes (int_ptr new_total_bytes) 
  {
    totalBytes_       = new_total_bytes;
    endTotalBytes()   = new_total_bytes;
  }

  // Mark this node as free 
  void markAsFree () 
  {
    int_ptr total_bytes = totalBytes_<0 ? -totalBytes_ : totalBytes_;
    setTotalBytes(-total_bytes); // negative total bytes
  }

  // Mark this node as inuse by the user
  void markAsInuse () 
  {
    int_ptr total_bytes = totalBytes_<0 ? -totalBytes_ : totalBytes_;
    setTotalBytes(+total_bytes); // positive total bytes
  }

  // Rare cases when you need the vector pool for allocating more
  // memory from the same vector pool.
  StreamingPool& streamingPool () const 
  { 
    return *pool_; 
  }

  // Take this node off of its list (the freelist)
  void unlink () 
  {
    prev_->next_ = next_;
    next_->prev_ = prev_;
  }

  // Put this on the given list (at the front)
  void link (SPInfo_* list)
  {
    next_ = list->next_;
    prev_ = list;
    list->next_->prev_ = this;
    list->next_ = this;
  }


  // Assuming this is a free node, see if it could satisfy a request
  // for memory of the given size and so that IF IT SPLIT, it's
  // split free node could also satisfy alignment constraints.
  // Returns the number of bytes this node would have to be split
  // into to keep the alignments proper.  If CANNOT satisfy request,
  // this returns 0
  int_ptr canSatisfy (int_ptr bytes_requested, int alignment)
  {
    // inuse node and free nodes have different overheads
    const size_t free_overhead = sizeof(SPInfo_);
    const size_t inuse_overhead = free_overhead-2*sizeof(next_); 
    
    // See if we split this free node into two nodes if enough space:
    // one inuse node (front) and one free node
    int_ptr bytes_in_inuse_node = bytes_requested+inuse_overhead;
    {
      // Make sure to correct so that split node will ALSO be aligned
      int_ptr where_next_free_node_would_start =
	((AVLP)this) + bytes_in_inuse_node + sizeof(totalBytes_) + sizeof(pool_);
      int_ptr remainder = where_next_free_node_would_start % alignment;
      if (remainder) {
	bytes_in_inuse_node += (alignment-remainder);
      }
      if (bytes_in_inuse_node>-totalBytes_) { // Not big enough...
	return 0;
      }
    }
    return bytes_in_inuse_node;
  }
   

  // Assuming there is enough space, split this node into an allocated
  // node and a free node.  Return the allocated node, and add the
  // leftovers to the freelist
  SPInfo_* splitNode (int_ptr bytes_requested, int alignment)
  {
    unlink(); // Take giant free node off of free list so we can split 
    
    // Try to split this free node into two nodes if enough space: one
    // inuse node (front) and one free node
    int_ptr bytes_in_inuse_node = canSatisfy(bytes_requested, alignment);
    int_ptr bytes_in_free_node = -totalBytes_ - bytes_in_inuse_node;
    if (bytes_in_free_node <= int_ptr(sizeof(SPInfo_))) {
      // Not enough for two nodes, just return this one as is
      this->markAsInuse();
      return this;
    }  

    // Inuse node
    SPInfo_* inuse_node = this;
    inuse_node->setTotalBytes(bytes_in_inuse_node);
    inuse_node->markAsInuse();
    // inuse_node->pool_ = pool_; // pool already set

    // Free node
    char* free_p = (char*) this;
    free_p += bytes_in_inuse_node;
    SPInfo_* free_node = (SPInfo_*)free_p; // compute where node is 
    free_node->setTotalBytes(bytes_in_free_node);
    free_node->markAsFree();
    free_node->pool_ = inuse_node->pool_;
    free_node->link(GetFreeList(free_node->pool_));

    return inuse_node;
  }

  // Coalesce the current (free) SpInfo with the next and return a
  // pointer the coalesced dude
  SPInfo_* coalesceWithNext (SPInfo_* next_free, bool unlink_myself=false)
  {
    // both totalBytes_ should be negative
    int_ptr total_number_of_bytes = totalBytes_ + next_free->totalBytes_;
    if (unlink_myself) {
      unlink();
    } else {
      next_free->unlink();
    }
    setTotalBytes(total_number_of_bytes);
    return this;
  }

  
 protected:
  // Only the StreamingPool can create these handles
  SPInfo_ (StreamingPool* front, 
	   SPInfo_* prev, SPInfo_* next) :
    pool_(front),
    prev_(prev), 
    next_(next)
  {  }
  
  // Destructor is protected so SPInfo_ instances may NOT be created
  // on the stack (or in global memory).  They must be allocated
  // WITHIN the Pool by the StreamingPool.  This is because the
  // instances are explicitly deleted by the pool when all references
  // to them have been eliminated, and we cannot have somebody else
  // calling the destructor first.
  ~SPInfo_ () { }

  // Similarly, make it so can't copy.  NO IMPL
  SPInfo_ (const SPInfo_& rhs);
  SPInfo_& operator= (const SPInfo_& rhs);
  
  // Total bytes in this node:  this includes
  // (1) total_bytes (2) front, prev and next (3) data in middle 
  // (4) total_bytes replicated at the end.  Why do we replicate   
  // total_bytes at the end? So we can coalesce by just looking at the 
  // next few bytes in the region next to us.
  int_ptr totalBytes_;
  StreamingPool* pool_;
  SPInfo_ *prev_, *next_; // When free node, these point to prev and next 
                          // on free node list.
                          // The prev, next fields ARE NOT IN an INUSE NODE!
  
  // This is ALMOST NEVER SET DIRECTLY ... USER DATA sits between the fields above 
  // and this field.  The 4 bytes at the END of the struct are usually
  // set by computing where this lays
  int_ptr totalBytesAtEnd_;

}; // OCSPInfo_


OC_END_NAMESPACE


#endif // OCSPINFO_H_

