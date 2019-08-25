#ifndef M2PMSTACK_H_
#define M2PMSTACK_H_

// Helper class for PM Context Stack: as objects are pushed and
// popped, this is the stack.  We have an extra piece of information
// so we can note pops_ and memoize correctly
template <class OBJ>
struct PStack_ { 
  PStack_ (const OBJ& v) :
    memo_number(-1),
    object(v)
  { }
  int memo_number;  // -1 means never memoized, otherwise offset into 
  OBJ  object;         // the actual object on the stack

}; // PStack_

// If we have a pointer to an OBJ in a PStack instance, return
// a pointer to the PStack instance.
template <class OBJ>
inline PStack_<OBJ>* ConvertFromObjPtrToPStackPtr (OBJ* obj_ptr)
{
  static PStack_<OBJ> dummy = OBJ();  // Just kept around for address arithmetic
  static char* dummy_front = (char*) &dummy;
  static char* dummy_inside = (char*) &dummy.object;
  static int diff = dummy_inside - dummy_front;
  
  char* obj = (char*) obj_ptr;
  char* pst = obj - diff;
  PStack_<OBJ>* ret_val = (PStack_<OBJ>*) pst;
  // NOTE: we could use offsetof, but I have found it to be non-portable
  // (i.e., some compilers don't support it).
  return ret_val;
}



// Helper class for memoization: Sometimes we store the values by
// pointer, sometimes by value.
template <class OBJ>
struct MemoInfo_ {
  MemoInfo_ (OBJ* o) : object_ptr(o==0? &object : o) { }

  OBJ* object_ptr;  // points to the actual full stack if item hasn't been pop:
                    // ALWAYS VALID: sometimes points to object (below),
                    // sometimes points to item on Pickle Stack

  OBJ object;       // Full object once been copied

  // If the objectPtr points the object inside, in the copy, it
  // continues to do that.  Other wise, the object_ptr points to
  // whatever it originally pointed to.
  MemoInfo_ (const MemoInfo_<OBJ>& rhs) :
    object(rhs.object)
  { if (&rhs.object == rhs.object_ptr) {
      object_ptr = & object;
    } else {
      object_ptr = rhs.object_ptr;
    }
  }
};  // MemoInfo_


// M2kand MITE Arrays DO NOT (currently) support SWAP!
// We have to work around this: WE NEED SWAP so that the
// new memos stay in place
#if !defined(OC_USE_OC)

template <class T>
void Swapping (T& lhs, T& rhs)
{
  T temp = lhs;
  lhs = rhs;
  rhs = temp;
}

template <class T>
class ArraySupportsSwapping : public Array<T> {
  public:

  // Don't use swap, because we will add it to MITE at some point
  void swapping (ArraySupportsSwapping<T>& rhs)
  {
    Swapping(this->length_, rhs.length_);
    Swapping(this->capac_, rhs.capac_);
    Swapping(this->useNewAndDelete_, rhs.useNewAndDelete_);
    Swapping(this->data_, rhs.data_);
  }

}; // ArraySupportsSwapping
#endif


template <class OBJ>
class PMStack_ {


 public:
   
  // Pop the stop of the PM Context stack and return it.  When we pop,
  // if this value had been memoized, then we need to copy the "full
  // copy" back to the memoize area.
  OBJ pop ()
  { 
    OBJ ret; // Return Value Optimization

    // When we pop, if this item has a memoization, we copy the memo back
    // over to memo area
    int index_of_last = stack_.length()-1;
    PStack_<OBJ>& p = stack_[index_of_last];
    ret = p.object;

    // If this memoized, copy memo back to memo area
    int memo_number = p.memo_number;
    if (memo_number!=-1) {
      MemoInfo_<OBJ>& m = memos_[memo_number]; 
      m.object = p.object;                    // copy full object
      m.object_ptr = &m.object;               // have pointer point HERE 
    }
    (void)stack_.removeLast(); 
    return ret;
  }

  // Multiple pops
  void pop (int how_many)
  {
    for (int ii=0; ii<how_many; ii++) {
      pop();
    } 
  }

  // Look at the top of the PM Stack: "peek" into the top (or a few
  // down) from the top of the stack.  0 means the top, -1 means 1
  // from the top, etc.
  OBJ& peek (int offset=0) 
  { 
    int index_of_entry = stack_.length()+offset-1;
    PStack_<OBJ>& p = stack_[index_of_entry];
    return p.object;
  }

  // Push an item on top of the PM Context stack.
  OBJ& push (const OBJ& v) 
  {
    const OBJ* vp = &v;

    // If a resize happens on the next append MOST MEMOS WILL BECOME
    // INVALID.  We have to copy the data and revalidate the memos.
    if (stack_.length() == stack_.capacity()) {
      // Make a copy, with bigger capacity so next append won't cause
      // resize.  We do this "manually" so we can have a copy of both
      // stacks at the same time to adjust memos correctly
      Array<PStack_<OBJ> > new_stack(stack_.length()*2); 
      int original_stack_len = stack_.length();
      for (int ii=0; ii<original_stack_len; ii++) {
	new_stack.append(stack_[ii]);
      }
      // Now, we have both stacks , so we can readjust memos
      PStack_<OBJ>* front_old_stack = &stack_[0];
      PStack_<OBJ>* front_new_stack = &new_stack[0];
      for (size_t jj=0; jj<memos_.length(); jj++) {
	MemoInfo_<OBJ>& m = memos_[jj];
	// Only have to readjust memo if points into stack
	// (not to itself)
	bool was_same_object_as_passed = (&v == m.object_ptr);
	if (m.object_ptr != &m.object) { 
	  // Point to new_stack object instead
	  PStack_<OBJ> *old_stack_place = ConvertFromObjPtrToPStackPtr(m.object_ptr);
	  int diff = old_stack_place - front_old_stack;
	  PStack_<OBJ> *new_stack_place = front_new_stack + diff;
	  m.object_ptr = &new_stack_place->object;
	  if (was_same_object_as_passed) {
	    vp = m.object_ptr;
	  }
	}
      }
      // Finally, install new bigger stack, where memos point to this
      // one! WE NEED SWAP so that the new memos stay in place
#if defined(OC_USE_OC)
      stack_.swap(new_stack);
#else 
      // M2kand MITE Arrays DO NOT (currently) support SWAP!
      // We have to work around this: 
      ArraySupportsSwapping<OBJ>* stackp=(ArraySupportsSwapping<OBJ>*)&stack_;
      ArraySupportsSwapping<OBJ>* new_stackp=(ArraySupportsSwapping<OBJ>*)&new_stack;
      stackp->swapping(*new_stackp);
#endif
    }
    // Big enough.
    stack_.append(PStack_<OBJ>(*vp)); 

    // Return element just pushed
    int len=stack_.length();
    PStack_<OBJ>& ret = stack_[len-1];
    return ret.object;
  }

  int length () const { return stack_.length(); }
  OBJ& operator[] (int index) 
  { 
    PStack_<OBJ>& r = stack_[index]; 
    return r.object; 
  }

  PStack_<OBJ>& operator() (int index) 
  { 
    PStack_<OBJ>& r = stack_[index]; 
    return r;
  }

  // Clear
  void clear () 
  { 
    stack_.clear(); 
    memos_.clear();
  }

  void memoGet (int_u4 memo_number)
  {
    //values_.push(memos_(memo_number));
    MemoInfo_<OBJ>& mm = memos_[memo_number]; 
    OBJ* shared = mm.object_ptr;
    // cout << "MEMO GOTTEN for #" << memo_number << " is " << *shared << endl;
    push(*shared);
  }

  // Uses top "peeked value" as the value to insert
  OBJ& memoPut (int_u4 memo_number)
  {
    int index_of_entry = stack_.length()-1;
    PStack_<OBJ>& p = stack_[index_of_entry];

    // cout << "MEMO PUT for #" << memo_number << " is "  << p.object << endl;

    p.memo_number = memo_number; // Record the memo number so pops correctly:

    // Associate this memo_number with the value:  Since memo numbers always
    // strictly increase, we can simply append to an array to assocate the
    // value with a number.  In other words: memo[memo_number] = &p.object
    while (int(memos_.length()) < int(memo_number)) {
      memos_.append(MemoInfo_<OBJ>(0));
    }
    memos_.append(MemoInfo_<OBJ>(&p.object)); 

    return p.object;
  }

 private:
  // The PM stack, as described by pickletools.  Every item on the
  // stack has "possibly" been memoized, which means when it is
  // popped, we need to copy the value out into the memo stack: that's
  // why we record the memo number along with the value (so we know
  // what its memo is).
  Array<PStack_<OBJ> > stack_; 

  // The memos stack, a reference to:
  // The "memo" (named values) as per pickletools.  More like a table
  // where we associate entries (memo numbers) with their Values.
  // While the values haven't been popped, the value is stored by
  // pointer.  When it is finally popped, its value is copied to the
  // memo stack.
  Array<MemoInfo_<OBJ> > memos_; 


}; // PMStack


#endif //  M2PMSTACK_H_
