
#include "timeconv.h"
#include <math.h>

PTOOLS_BEGIN_NAMESPACE

// Constants
const int_u2 TimeConv::DefaultPrecision = 3;
const int_u2 TimeConv::MaxPrecision = 11;
const TimeConv::Clock_t TimeConv::QuantaPerSecond = 4000000000UL; 
const TimeConv::Clock_t TimeConv::QuantaPerNanosecond = 4L;
const TimeConv::Clock_t TimeConv::QuantaPerMicrosecond = 4000L;
const int_u4 TimeConv::SecondsPerDay = 24*60*60;
const int_u4 TimeConv::SecondsPerHour = 60*60;
const int_u4 TimeConv::SecondsPerMinute = 60;

// From M2Date
const int_u4 TimeConv::juloffset = 2433282;


    
// The number of seconds
TimeConv::Second_t TimeConv::second () const
{
  // This is trickier than it looks.  We don't want to cast quanta_ to
  // double, because we can lose many bits of precison.  However, if
  // we operate only on the fractional part of the second, we'll
  // probably be safe.  
  int_u2 isec = int_u2((quanta_ / QuantaPerSecond) % SecondsPerMinute);
  Second_t frac_sec = quanta_ % QuantaPerSecond;
  frac_sec /= QuantaPerSecond;
  
  return isec + frac_sec;
}                                       // TimeConv::second 

// The number of hours
TimeConv::Hour_t TimeConv::hour () const
{
  return TimeConv::Hour_t(((quanta_ / QuantaPerSecond) % SecondsPerDay)
			  / SecondsPerHour);
}                                       // TimeConv::hour 

// The number of minutes
TimeConv::Minute_t TimeConv::minute () const
{
  return TimeConv::Minute_t( (((quanta_ / QuantaPerSecond) % SecondsPerDay)
			      % SecondsPerHour) / SecondsPerMinute);
}                                       // TimeConv::minute 

PTOOLS_END_NAMESPACE

#include <iomanip>
using std::setw;
using std::setprecision;

PTOOLS_BEGIN_NAMESPACE

// Helper routine to write out
void TimeConv::writeTOD_ (ostream& os, int_u2 precision) const
{
  os << setw(2) << hour() << ":" << setw(2) << minute() << ":";
  if (precision > TimeConv::MaxPrecision) {
    precision = TimeConv::MaxPrecision;
  }
  
  if (precision > 0) {
    os << setprecision(precision) << setw(precision + 3) << second();
  } else {
    // We will take the floor of the seconds so that we can guarentee
    // a seconds value of 0-59.
    os << setw(2) << int_u2(floor(second()));
  }
}                                       // TimeConv::writeTOD_



// Really what we want: the string version of the M2Time (from int_u8)
ostream& TimeConv::prettyPrint (ostream& os) const
{
  // Get date as julian: code from TimeConv/M2Date
  int_u4 daycount = (long)(quanta_ / QuantaPerSecond / SecondsPerDay);
  Julian_t julnum = daycount + 1;
  
  os.setf(ios::fixed);
  os << std::setfill('0');
  
  Month_t month; Day_t day; Year_t year;
  monthDayYear_ (julnum, month, day, year);
  os << std::setfill('0') << std::setw(4) << year << ":"
     << std::setw(2) << month << ":" << std::setw(2) << day << "::";
  
  writeTOD_(os, TimeConv::MaxPrecision);
  return os;
}



// Convert a modified Julian day number to its corresponding
// Gregorian calendar date.  Algorithm 199 from Communications of
// the ACM, Volume 6, No. 8, (Aug. 1963), p. 444.  Gregorian
// calendar started on Sep. 14, 1752.  This function not valid
// before that.
void TimeConv::monthDayYear_ (Julian_t julnum_,
			      Month_t& month, Day_t& day, Year_t& year) const
{
  Julian_t j = julnum_ + TimeConv::juloffset - 1721119;
  unsigned long y = ((j<<2) - 1) / 146097;
  j = (j<<2) - 1 - 146097*y;
  unsigned long d = j>>2;
  j = ((d<<2) + 3) / 1461;
  d = (d<<2) + 3 - 1461*j;
  d = (d + 4)>>2;
  unsigned long m = (5*d - 3)/153;
  d = 5*d - 3 - 153*m;
  d = (d + 5)/5;
  y = 100*y + j;
  if (m < 10)
    m += 3;
  else {
    m -= 9;
    y++;
  }
  month = (Month_t)m;
  day   = (Day_t)d;
  year  = (Year_t)y;
}

PTOOLS_END_NAMESPACE
