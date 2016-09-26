#ifndef OCTUP_H_

// Like a python Tuple: contains a constant-length list of
// items.  A tuple is very similar to a list in a lot of ways,
// but the way they are constructed with makes them feel more like
// Python tuples.  For example:
//
//   Tup empty;     // empty tuple
//   Tup one(1.1);    // one entry: a real_8 
//   Tup another(1);  // one entry: an int_4 
//
//   Tup pair(1, 2.2); // two entries: an int_4 and real_8
//
//   Tup t(1, 2.2, "three", Tab("{ 'a':1 }"), Arr("[1,2,3]"));
//   int_u4 ii = t[0];
//   real_8 rr = t[1];
//   string ss = t[2];
//   Tab&   tt = t[3];
//   Arr&   aa = t[4];
//   cout << t << endl;           // (1, 2.2, "three", {'a':1}, [1,2,3])
//   cout << t.length() << endl;  // 5 
//   cerr << t[6];                // Exception!  Only 5 in this tuple
//
// 
// Things in tuples are constructed as is: they don't have a single
// string literal syntax like Tabs and Arrs.
//
//   Tup t1("(1,2,3)") ;       // t1 is a tuple with 1-element: a string
//
// You can get the same effect using Eval:
//   Tup t2 = Eval("(1,2,3")); // t2 is a tuple with 3 elements:int * 3
//
// Using Tups with Vals is very similar to Tab& and Arr&: you just
// ask for tuple out.
//
//   Val v = Tup(1,2,3);
//   Tup& t = v;          // Get a reference to the Tup inside
//   Tup copy = v;        // Get my own copy

class Tup {
 public:

  // Allow the user to specify a tuple of any size desired: the number
  // of entries in the constructor is how big the tuple is
  Tup () 
    : a_(0) { }
  Tup (const Val& a) 
    : a_(1) { a_.append(a); }
  Tup (const Val& a, const Val& b) 
    : a_(2) { a_.append(a); a_append(b); }
  Tup (const Val& a, const Val& b, const Val& c) 
    : a_(3) { a_.append(a); a_append(b); a_.append(c); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d) 
    : a_(4) { a_.append(a); a_append(b); a_.append(c); a_.append(d); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& d) 
    : a_(5) { a_.append(a); a_append(b); a_.append(c); a_.append(d); a_.append(e); }
  
  // Like the Array operations inspection operations
  inline size_t length () const { return a_.length(); }
  inline size_t entries () const { return a_.entries(); }
  inline bool   contains (const Val& v) const { return a_.contains(v); }

  // Like the Array operations: can't do non-const operations on Tup,
  // just like Python tuple
  void   append (const Val& v) 
  { throw runtime_error("Can't do appendStr on Tup, use Arr instead"); }
  void   appendStr (const Val& v) 
  { throw runtime_error("Can't do append on Tup, use Arr instead"); }
  bool   remove (const Val& v) 
  { throw runtime_error("Can't do remove on Tup, use Arr instead"); }

  // Helper functions
  
  // Swap in O(1) time.
  inline void swap (Tup& rhs) { a_->swap(rhs.a_);}

  // Pretty print the table (with indentation)
  OC_INLINE void prettyPrint (ostream& os, int starting_indent=0, 
			      int additive_indent=4) const;
  OC_INLINE ostream& prettyPrintHelper_ (ostream& os, int indent, bool pretty=true, int additive_indent=4) const;

  // Give access to allocator
  inline Allocator* allocator () const { return allocator_; }

  // If the key is in the OTab, then return the corresponding
  // value: if it's not, then return the given default value.
  // This does NOT change the OTab underneath.  Note that this
  // returns a Val (not a Val reference) since it's possible
  // to return a default value, which may not anywhere in particular.
  OC_INLINE Val get (const Val& key, const Val& def=None) const;


  // Like the array operarions
  const Val& operator[] const (size_t i) { return a_[i]; }
  const Val& operator() const (size_t i) { return a_(i); }
        Val& operator[] (size_t i) { return a_[i]; }
        Val& operator() (size_t i) { return a_(i); }


 protected:
  // Implementation: just an array!
  Array<Val> a_;
  
}; // Tup

// Comparison
inline bool operator== (const Tup& lhs, const Tup& rhs)
{ return lhs.a_ == rhs.a_; }
inline bool operator< (const Tup& lhs, const Tup& rhs)
{ return lhs.a_ < rhs.a_; }
inline bool operator<= (const Tup& lhs, const Tup& rhs)
{ return lhs.a_ <= rhs.a_; }
inline bool operator> (const Tup& lhs, const Tup& rhs)
{ return lhs.a_ > rhs.a_; }
inline bool operator>= (const Tup& lhs, const Tup& rhs)
{ return lhs.a_ >= rhs.a_; }

// stream
OC_INLINE ostream& operator<< (ostream& os, const Tup& t)
{
  os << "(";
  const int len = int(t.length());
  for (int ii=0; ii<int(t.length()); ii++) {
    os << t[ii];
    if (ii==len-1) {
      os << ")";
    } else {
      os << ", ";
    }
  }
}

OC_INLINE ostream& Tup::prettyPrintHelper_ (ostream& os, int indent, 
					    bool pretty, int indent_additive) const
{
  // Base case, empty 
  if (entries()==0) return os << "( )"; 
  
  // Recursive case
  os << "(";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  int ent = entries();
  for (int ii=0; ii<ent; ++ii) {
    const Val& value = (*this)[ii];
    
    if (pretty) indentOut_(os, indent+indent_additive);
    
    // For most values, use default output method
    switch (value.tag) {
    case 'a': {
      OCString* ap = (OCString*)&value.u.a; 
      os << PyImage(*ap, true);
      break; 
    }
    case 't': { 
      Tab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'o': { 
      OTab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'u': { 
      Tup& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'n': {
      if (value.subtype=='Z') {
	Arr& arr = value;
	arr.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			       pretty, indent_additive);
	break;
      } // else fall thru for other array types
    }
    default: os << value; break;
    }
    
    if (entries()>1 && ii!=int(entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << ")";
}

#define OCTUP_H_
#endif // OCTUP_H_
