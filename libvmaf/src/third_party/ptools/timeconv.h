#ifndef TIMECONV_H_
#define TIMECONV_H_

// For converting int_u8 (M2Time) to a string format.  We could use
// UNIX strftime, but we want to make sure if M2k is sending us an
// M2Time, we do the *exact* same conversion it does for consistency.
// This is really only useful in M2k binary serialization in
// converting the int_u8 to a (what M2k calls TimeStandard) string.

// For stringizing a TimeConv, either:
// out << TimeConv(some_int_u8) << endl;
//   or
// string s = Stringize(TimeConv(some_int_u8));  
// NOTE:  Stringize uses operator>> to do the conversion for us.

// ///// Includes
#include "ocport.h"

PTOOLS_BEGIN_NAMESPACE

// A helper class for encapsulating all the necessary M2Time conversion
// from int_u8 to string or stream (see above).
class TimeConv { 
  
 public:
  typedef int_u2 Day_t;         // day of month
  typedef int_u2 Month_t;       // month
  typedef int_u2 Year_t;        // year
  typedef int_u2 Hour_t;
  typedef int_u2 Minute_t;
  typedef real_8 Second_t;
  
  typedef int_u4 Julian_t;      // Julian day



  // We must have a numeric type which names the units in which
  // quanta are represented.  Rather than put Quant1950ns_t or
  // Quanta_qns_t or Quanta_picoSec_t or whatever throughout the
  // baseline, the type Quanta_t is to always be equivalent to
  // whatever type quanta are stored in.  We'll keep Quanta1950ns_t
  // and Clock_t for backwards compatibility.
  typedef int_u8 Quanta1950ns_t;
  typedef Quanta1950ns_t Clock_t;
  typedef int_u8 Quanta_t;

  // ///// Constants  
  static const Quanta_t QuantaPerSecond;
  static const Clock_t  QuantaPerMicrosecond;
  static const Quanta_t QuantaPerNanosecond;
  static const int_u4   SecondsPerDay;
  static const int_u4   SecondsPerHour;
  static const int_u4   SecondsPerMinute;
  static const int_u2   DefaultPrecision;
  static const int_u2   MaxPrecision;
  static const Julian_t juloffset;

  // Construct a TimeConv (essentially a parred-down M2Time)
  // so you can convert it to a string.
  inline TimeConv (int_u8 value) : 
    quanta_(value) 
  { }

  // The number of seconds, hours, min
  Second_t second () const;
  Hour_t hour () const;
  Minute_t minute () const;

  // Print the TimeConv to the given stream (Stringize will
  // automatically use this for us).
  ostream& prettyPrint (ostream& out) const;

 protected:
  Quanta_t quanta_;

  // Month, year, date, given julian 
  void monthDayYear_ (Julian_t julnum_,
		      Month_t& month, Day_t& day, Year_t& year) const;
  
  // Helper to write Time Of Day
  void writeTOD_ (ostream& out, int_u2 precision) const;

}; // TimeConv


// ///// Global Functions

// Output TimeConv to stream so we can see what that int_u8 actually means!
inline ostream& operator<< (ostream& os, const TimeConv& tc) 
{
  return tc.prettyPrint(os);
}

PTOOLS_END_NAMESPACE

#endif // TimeConv
