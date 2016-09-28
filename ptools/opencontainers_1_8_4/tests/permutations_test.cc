//

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "ocarray.h"
#include "ocpermutations.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// ///////////////////////////////////////////// The PermutationsTest Class

class PermutationsTest {
    
  public:

    // ///// Methods

    int tests ();

}; // PermutationsTest


// ///////////////////////////////////////////// Globals

int_u8 fact (int_u8 n)
{
  int_u8 ret = 1;
  for (; n!=0; n--) 
    ret *= n;
  return ret;
}

// ///////////////////////////////////////////// PermutationsSimple

// This class computes permutations for you with little extra space
// (many schemes require you to generate all permutations which eats
// up excessive space: The extra space needed by this algorithm is
// about O(n^2) as opposed to O(n!) which you expect for
// permutations).

// All permutations of 1..n are generated, one at a time.  For example:

// PermutationsSimple p(5); // Give me all permutations from 1..5
// while (p.next()) {
//   cerr << p.currentPermutation() << endl;
// }

// Implementation note: This is essentially just the recursive version
// of permute unwrapped so that it's n levels of recursion are
// represented by elements of an array: It's quite fast (although not
// as fast as the baseline version).  There is a lot of confidence in
// this code, so it's good to compare against.

class PermutationsSimple {

    // A Helper class for computing permutations.
    
    class Perms_ {
	
	enum { PERMS_LOOKASIDE_LENGTH = 16 };

      public:
	
	// Create a permutation generator for permutations of length
	// n. The link is the generator we point to which will give us
	// the next currentPermutation_ of length n_-1.
	
	Perms_ (int n, Perms_* link) :
	  n_(n),
	  currentPermutation_(lookasideBuffer_),
	  ii_(n), // This will cause the next to get the next permutation
	  link_(link)
	{
	  if (n_>PERMS_LOOKASIDE_LENGTH) 
	    currentPermutation_ = new int[n-1];
	}
	
	
	// Clean up
	
	~Perms_ ()
	{
	  if (n_>PERMS_LOOKASIDE_LENGTH) {
	    delete [] currentPermutation_;
	  }
	}
	
	
	// Copy the next permutation into given buffer; the next
	// permutation is of length n.
	
	bool next (int* result)
	{
	  // The next result is of length n_: cache a few items for
	  // speed.
	  int n = n_;
	  int* cp = currentPermutation_;
	  int ii = up_();

	  // Whoops!  No more upping! All done.
	  if (ii==-1) 
	    return false;
	  
	  // Copy all of the current permutation, but put
	  // the n_ at the ii_ position
	  int k1, k2;
	  for (k1=0; k1<ii; k1++) 
	    result[k1] = cp[k1];
	  result[ii] = n;
	  for (k2=ii+1; k2<n; k2++)
	    result[k2] = cp[k2-1];
	  
	  // All done
	  return true;
	}

	// Inspector

	int* currentPermutation () const { return currentPermutation_; }

	
      protected:
	
	// ///// Data Members
	
	// This class is generating all permutations of this length
	
	int n_;
	
	// The current permutation of length n_-1 (To create
	// permutations of length n_, we distribute "n" over all
	// n_ positions of some permutation of length n_-1).
	
	int* currentPermutation_;
	
	
	// Where we are currently in the distribution: We are going to
	// distribute n_ at ii the next time "next" is called.
	
	int ii_;
	
	
	// Someone must supply us with the next "currentPermutation_"
	// when we exhaust the current one.  It is NOT adopted memory:
	// We use NULL to indicate the end of the chain
	
	Perms_* link_;

	
	// Because permutations are small, we have a lookaside buffer
	// for speed (most of the time, we can go to the lookaside
	// buffer).

	int lookasideBuffer_[PERMS_LOOKASIDE_LENGTH];

	
	// ///// Methods
	
	// This ups where we are going to distribute the next "n" to,
	// and if we reach the end, it updates the
	// currentPermutation_.  This does so in a pre_increment kind
	// of fashion.
	
	int up_ ()
	{
	  // Special case for very last one (when n==1)
	  if (!link_) {

	    // It generated it's one item, all done!
	    if (ii_!=n_)
	      return -1;

	    // Otherwise the item will be generated with ii_=0
	    ii_ = 0;
	    return ii_;
	  }

	  // Normal case for every one else
	  if (++ii_ >= n_) {
	    ii_ = (link_->next(currentPermutation_)) ? 0 : -1;
	  }
	  return ii_;
	}

      private:

	// Prohibit copying: HACK: because Array::resize needs to know
	// about it, we need the Copy COnstructor even though we
	// assure that the code never gets called.  
	// Perms_(const Perms_& x); 
        // Perms_& operator= (const Perms_& x);

    }; // Perms_


  public:

    // ///// Methods

    // Constructor.  Will generate all permutations from 1..n (all n
    // factorial of them).  Until the "next" method is called, the
    // current permutation is invalid.

    PermutationsSimple (int n);


    // Destructor

    ~PermutationsSimple ();


    // Advance to the next permutation: This methods needs to be
    // called before currentPermutation can be called.
    
    bool next ();


    // Give me read access to the current permutation: You really
    // shouldn't be modifying this, but sometimes the const qualifier
    // is too strict.

    int* currentPermutation () const;


  protected:

    // ///// Data Members

    // We keep a chain of generators, one for each stage: permutations
    // of length n, length n-1, ..., length 1.

    Array<Perms_> generator_;


    // Where the result goes

    int* result_;


}; // PermutationsSimple




// ///////////////////////////////////////////// PermutationsSimple Methods

PermutationsSimple::PermutationsSimple (int n) :
  generator_(n),
  result_(0)
{
  // Sanity check
  if (n<0) {
    throw out_of_range("Illegal construction of instance PermutationsSimple");
  }
  
  // Handle n==0 by simply not having and Perms_
  if (n==0)
    return;
  
  // Where the result (can) go.
  result_ = new int[n];

  // No constructors have been called yet ..  we are going to
  // construct each one in place (carefully!).
  generator_.expandTo(n);
  PermutationsSimple::Perms_* p = (PermutationsSimple::Perms_*)generator_.data();

  // Construct: generator_[0] is for perms of length 1, generator_[1]
  // is for perms of length 2, ... generator_[n-1] for perms of length
  // n.  Note that we have to construct generator_[0] first because it
  // doesn't link to any others.
  new (p) PermutationsSimple::Perms_(1, NULL); // In place constructor
  for (int ii=1; ii<n; ii++) {
    p++;
    new (p) PermutationsSimple::Perms_(ii+1, p-1); // In place constructor
  }

}



PermutationsSimple::~PermutationsSimple ()
{
  // Although we constructed them in-place, they can be reclaimed via
  // normal Destructor procedures, so we don't need to do anything
  // here.

  delete [] result_;
}



bool PermutationsSimple::next ()
{
  // n==0, never return true (as we never generate anything).
  if (generator_.length()<=0)
    return false;
  
  const size_t last = generator_.length()-1;
  return generator_[last].next(result_);
}



int* PermutationsSimple::currentPermutation () const
{
  // Note this is a copy and internally, the current Permutations are
  // always copied into each others buffers with extra copies always
  // being made.
  return result_;
}



// ///////////////////////////////////////////// PermutationsTest Methods

int PermutationsTest::tests()
{
  cout << "** Test of Permutations **" << endl;
  {
    for (int n=0; n<6; n++) {
      cout << "-> Permutations of " << n << endl;
      Permutations p(n);
      while (p.next()) {
	const int* perm = p.currentPermutation();
	for (int ii=0; ii<n; ii++) {
	  cout << perm[ii] << " ";
	}
	cout << endl;
      }
    }
  }


  cout << "**Test of PermutationsT **" << endl;
  {
    const char* names[] = { "one", "two", "three", "four", "five", "six" };

    for (int n=0; n<6; n++) {
      cout << "-> Permutations of " << n << endl;
      Array<string> a(n); 
      for (int  uu=0; uu<n; uu++)
	a.append(string(names[uu]));
      PermutationsT<string> p(a);
      while (p.next()) {
	cout <<  p.currentPermutation() << endl;
      }
    }
  }



  cout << "** Hard Test:  compare two permutation generators ** " << endl;

  for (int n=0; n<8; n++) {

    cout << "Permutations of length " << n << endl;

    // Set up "against" with all permutations, and a flag array
    // "there" to see if it's been hit.
    PermutationsSimple ps(n);
    Array<Array<int> > against(n*n);
    while (ps.next()) {
      const int* cp = ps.currentPermutation();
      Array<int> newone(n);

      
      for (int ii=0; ii<n; ii++)
	newone.append(cp[ii]);
      // cout << " BASE: " << newone << endl;
      against.append(newone);
    }
    // cout << "against.length() = " << against.length() << endl;
    Array<int> there(against.length());
    for (int ii=0; ii<int(against.length()); ii++) {
      there.append(0);
    }
    // cout << "there.length() = " << there.length() << endl;
    // cout << "there = " << there << endl;

    // Now, using the baseline permutation tester, create and compare
    Permutations p(n);
    while (p.next()) {
      Array<int> current_answer(n);
      const int* cp = p.currentPermutation();
      for (int dd=0; dd<n; dd++) {
	current_answer.append(cp[dd]);
      }
      // cout << "current = " << current_answer << endl;
      
      int jj;
      bool found = false;
      for (jj=0; jj<int(against.length()); jj++) {
	
	// cout << "jj = " << jj << " -> " << against[jj] << endl;
	
	bool equal_stuff = true;
	for (int ee=0; ee<n; ee++) {
	  if (current_answer[ee] != against[jj][ee]) {
	    equal_stuff = false;
	    break;
	  }
	}
	if (equal_stuff) {
	  // cout << "Found: " << current_answer << " at index " << jj << endl;
	  if (there[jj] == true) {
	    cout << "ERROR: The " << current_answer 
		 << " was already at index " << jj << endl;
	  }
	  there[jj] = 1;
	  found = true;
	  break;
	  
	  
	}
      }
      if (!found) {
	cout << "ERROR:  Did NOT find " << current_answer << endl;
      } else {
	// cout << "OKAY for permutations of length " << n << endl;
      }
    }

    // Double check that everything was tagged
    for (int ff=0; ff<n; ff++) {
      if (there[ff]==0) 
	cout << "ERROR:  misssing : " << against[ff] << "  Not there at index " << ff << endl;
    }
  }



  // Test that we can seed it.

  {
    cout << " ** Seed Simple test first:  Try the first six, as well \n"
      "** as a few after (to see if if returns round)." << endl;
    
    Permutations pp(3);
    for (int ww=0; ww<13; ww++) {
      pp.seed(ww);
      for (int yy=0; yy<3; yy++)
	cout << pp.currentPermutation()[yy] << " ";
      cout << endl;
    }
    
    for (int ss=0; ss<11; ss++) {

      cout << " ** Testing to see if all seeds match a normal generation for length = " << ss << "**" << endl;
      Permutations p(ss);
      
      Permutations p1(ss);

      int_u8 length = fact(ss);
      if (ss==0) length = 0;
      cout << " ... expecting " << length << endl;
      for (int_u8 st=0; st<length; st++) {
	
	if (!p1.next()) {
	  cout << "ERROR: Early return" << endl;
	  break;
	}
	

        p.seed(st);
        for (int yy=0; yy<ss; yy++) {
	  if (p1.currentPermutation()[yy] != p.currentPermutation()[yy]) {
	    cout << "ERROR: Permutation: " << st << " has a mismatch" << endl;
	  }
	  // cout << p.currentPermutation()[yy] << " "; 
	}
        // cout << endl;

	// compare
	
      }
      if (p1.next()) {
	cout << "ERROR:  Expecting p1 to be done at that point" << endl;
      }
      if (p.next()) {
	cout << "ERROR:  Expecting p to be done at that point" << endl;
      }
    }

   
  }
  
  return 0;
}



int main ()
{
  PermutationsTest t;
  return t.tests();
  return 0;
}



