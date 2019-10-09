#ifndef OCMOVE_H_
#define OCMOVE_H_

#include "ocport.h"
#include "string.h" // for memcpy

OC_BEGIN_NAMESPACE

/*********************************************************
 *** Some of the technqiues used here work in more modern
 *** compilers, but even slightly older RedHat 5.4 and
 *** a modern Intel 11 had trouble with the SFINAE techniques
 *** below.  We keep these around for when SFINAE is more
 *** mainstream, or perhaps more portable.  In the meantime,
 *** we use overloading on MoveArray to solve the problem.


// Some structure that allows us to detect whether
// a type (a) is primitive or (b) has the move method
enum Movable_e { NO_SPECIAL_MOVE=0, HAS_SWAP=0x1, HAS_MOVE=0x3, HAS_MOVEARRAY=0x7 };

template <typename Klass>
struct Moving {

  // For moveArray method:  uses ptr to member functions and SFINAE
  template<class U> static char testMA( char(*)[sizeof(&U::moveArray)]);
  template<class U> static long testMA(...);  // if no move
  static const int has_moveArray_method = 7*(sizeof(testMA<Klass>(0))==sizeof(char));

  // For move method:  uses ptr to member functions and SFINAE
  template<class U> static char testM( char(*)[sizeof(&U::move)]);
  template<class U> static long testM(...);  // if no move
  static const int has_move_method = 3*(sizeof(testM<Klass>(0))==sizeof(char));

  // For swap method:  uses ptr to member functions and SFINAE
  template<class U> static char testS( char(*)[sizeof(&U::swap)]);
  template<class U> static long testS(...);  // if no move
  static const int has_swap_method = sizeof(testS<Klass>(0))==sizeof(char);


  static const Movable_e moves = Movable_e(has_swap_method | has_move_method);
  static const Movable_e movesArray = Movable_e(has_swap_method | has_move_method | has_moveArray_method);
};

// This section gives us an optimized Move: if the class
// has a move method, it uses that, otherwise it copies over and
// destructs.  Move hasa curious semantics: when it is starts:
// the from is constructed and to is "raw memory": when it finishes,
// from is "raw memory" and to is constructed.

// Modern C++ Design: allows us to switch on types
template <int v> struct Int2Type { enum {value=v}; };

// Moving supported
template <typename T> 
inline void Move (T* from, T* to, Int2Type<HAS_MOVE>) 
{ 
  //cerr << "SINGLE supports move" << endl;
  from->move(to); 
}

// No move, but swap: swap is usually at least linear time
template <typename T> 
inline void Move (T* from, T* to, Int2Type<HAS_SWAP>) 
{ 
  //cerr << "SINGLE supports move" << endl;
  new (to) T();
  from->swap(*to); // usually linear time
  from->~T();
}

// Moving not supported, have to make copy then destruct
template <typename T> 
inline void Move (T* from, T* to, Int2Type<NO_SPECIAL_MOVE>) 
{ 
  //cerr << "SINGLE NO supports move"  << endl;
  new (to) T(*from); 
  from->~T(); 
}

// Generic move that does the right thing:  Call this and
// it will handle the optimized way
template <typename T>
inline void Move (T* from, T* to)
{
  Move(from, to, Int2Type<Moving<T>::moves>());
}

// Moving supported
template <typename T> 
inline void MoveArray (T* from, T* to, int len, Int2Type<HAS_MOVEARRAY>) 
{ from->moveArray(to, len); }

template <typename T> 
inline void MoveArray (T* from, T* to, int len, Int2Type<HAS_MOVE>) 
{ 
  //cerr << "supports move" << endl;
  for (int ii=0; ii<len; ii++) {
    from[ii].move(&to[ii]);       // linear time move (hopefully)
  }
}

// Don't have move, but have swap: usually better to use swap if they 
// have it
template <typename T> 
inline void MoveArray (T* from, T* to, int len, Int2Type<HAS_SWAP>) 
{ 
  //cerr << "supports swap" << endl;
  for (int ii=0; ii<len; ii++) {
    new (&to[ii]) T();
    from[ii].swap(to[ii]);  // usually O(1) move
    from[ii].~T();
  }
}

// Moving nor swap supported, have to make copy then destruct
template <typename T> 
inline void MoveArray (T* from, T* to, int len, Int2Type<NO_SPECIAL_MOVE>) 
{ 
  //cerr << "NO supports move"  << endl;
  for (int ii=0; ii<len; ii++) {
    new (&to[ii]) T(from[ii]);  // Potentially fill linear-time copy
    from[ii].~T(); 
  }
}


// Generic move that does the right thing:  Call this and
// it will handle the optimized way
template <typename T>
inline void MoveArray (T* from, T* to, int len)
{
  MoveArray(from, to, len, Int2Type<Moving<T>::movesArray>());
}
*/


// Generic MoveArray: has to run destructors on from
template <typename T> 
inline bool MoveArray (T* from, T* to, int len) 
{ 
  for (int ii=0; ii<len; ii++) {
    new (&to[ii]) T(from[ii]);  // Potentially full linear-time copy
    from[ii].~T(); 
  }
  return false;  // destructors run
}

#define MOVEARRAYPOD(TT) \
  template <> inline bool MoveArray<TT> (TT* from, TT* to, int len) { \
  memcpy(to, from, sizeof(TT)*len); return false; }

MOVEARRAYPOD(char);
MOVEARRAYPOD(int_1);
MOVEARRAYPOD(int_u1);
MOVEARRAYPOD(int_2);
MOVEARRAYPOD(int_u2);
MOVEARRAYPOD(int_4);
MOVEARRAYPOD(int_u4);
MOVEARRAYPOD(int_8);
MOVEARRAYPOD(int_u8);
//MOVEARRAYPOD(complex_8);  // has to be right after defined
//MOVEARRAYPOD(complex_16); // has to be right after defined

OC_END_NAMESPACE

#endif // OCMOVE_H_

