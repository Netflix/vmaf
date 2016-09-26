//
//
// ///// Authors: Jeff Schoen, Scott Schoen

// Low-level data representation conversion routines.


// ///////////////////////////////////////////// Include Files

#include "m2convertrep.h"


PTOOLS_BEGIN_NAMESPACE

// ///////////////////////////////////////////// Enum Type Methods


/*
M2ENUM_STREAM_OPS_DEFS(MachineRep_e)

M2ALPHABET_BEGIN_MAP(MachineRep_e)

  AlphabetMap("IEEE", MachineRep_IEEE);
  AlphabetMap("EEEI", MachineRep_EEEI);
  AlphabetMap("VAX", MachineRep_VAX);
  AlphabetMap("CRAY", MachineRep_CRAY);

  AcceptableMap("I", MachineRep_IEEE);
  AcceptableMap("E", MachineRep_EEEI);
  AcceptableMap("V", MachineRep_VAX);
  AcceptableMap("C", MachineRep_CRAY);

M2ALPHABET_END_MAP(MachineRep_e)
*/

// ///////////////////////////////////////////// Byte Swap Routines


static void Swap2 (void* inbuf, Size n)
{
  Index j = n * 2;
  unsigned char* buf = reinterpret_cast(unsigned char*, inbuf);
  unsigned char b;

  for (Index i = 0; i < j; i += 2) {
    b = buf[i];
    buf[i] = buf[i + 1];
    buf[i + 1] = b;
  }
} // Swap2



static void Swap4 (void* inbuf, Size n)
{
  Index j = n * 4;
  unsigned char* buf = reinterpret_cast(unsigned char*, inbuf);
  unsigned char b;

  for (Index i = 0; i < j; i += 4) {
    b = buf[i];
    buf[i] = buf[i + 3];
    buf[i + 3] = b;
    b = buf[i + 1];
    buf[i + 1] = buf[i + 2];
    buf[i + 2] = b;
  }
} // Swap4



static void Swap8 (void* inbuf, Size n)
{
  Index j = n * 8;
  unsigned char* buf = reinterpret_cast(unsigned char*, inbuf);
  unsigned char b;

  for (Index i = 0; i < j; i += 8) {
    b = buf[i];
    buf[i] = buf[i + 7];
    buf[i + 7] = b;
    b = buf[i + 1];
    buf[i + 1] = buf[i + 6];
    buf[i + 6] = b;
    b = buf[i + 2];
    buf[i + 2] = buf[i + 5];
    buf[i + 5] = b;
    b = buf[i + 3];
    buf[i + 3] = buf[i + 4];
    buf[i + 4] = b;
  }
} // Swap8




// ///////////////////////////////////////////// Global MachineRep_e functions

MachineRep_e DecodeMachineRep (const string& str)
{
  switch (str[(size_t) 0]) {
    case 'I':
      return MachineRep_IEEE;
      M2BOGUS_BREAK;
    case 'E':
      return MachineRep_EEEI;
      M2BOGUS_BREAK;
    case 'V':
      return MachineRep_VAX;
      M2BOGUS_BREAK;
    case 'C':
      return MachineRep_CRAY;
      M2BOGUS_BREAK;
    default:
      return MachineRep_UNDEFINED;
      M2BOGUS_BREAK;
  }	// switch off of 1st character

}					// DecodeMachineRep



string EncodeMachineRep (MachineRep_e rep)
{
  switch (rep) {
    case MachineRep_IEEE:
      return "IEEE";
      M2BOGUS_BREAK;
    case MachineRep_EEEI:
      return "EEEI";
      M2BOGUS_BREAK;
    case MachineRep_VAX:
      return "VAX";
      M2BOGUS_BREAK;
    case MachineRep_CRAY:
      return "CRAY";
      M2BOGUS_BREAK;
    default:
      return "Undefined";
      M2BOGUS_BREAK;
  }	// switch
}					// EncodeMachineRep




// ///////////////////////////////////// Floating Point Conversion Routines

void f_ieee2eeei (real_4 *buffer, Size n)
{
  Swap4(buffer, n);
}

void f_eeei2ieee (real_4 *buffer, Size n)
{
  Swap4(buffer, n);
}




// ///////////////////////////////////////////// VAX/EEEI Conversions

static void f_vax2ieee (real_4 *buffer, Size n)
{
//  cerr << "f_vax2ieee" << endl;
#if CNVX_
  Size nbytes = n * sizeof(float);
  cvt$vaxfToIEEE(buffer, buffer, &nbytes);
#else
  int i;

  // 1st, word swap to match local machine.
#ifdef M2_IEEE_
  Swap4(buffer, n);
#elif defined(M2_EEEI_)
#elif M2_VAX_
#endif

  int_u4 *u4 =(int_u4*)buffer;

//  int sign;
  int exponent;
  int mantissa;

  int nnn = n;
  for (i = 0; i < nnn; i++) {
    //int_u4 sign_mask = 1 << 15;
    int_u4 exponent_mask = (0xff << 7);


    // just to show where the sign is
    //    sign = u4[i] & sign_mask;
    //    sign = sign >> 15;
    mantissa = u4[i] & 0x7f;
    mantissa = mantissa << 16;
    mantissa = mantissa | (u4[i] & 0xffff0000) >> 16;
    exponent = u4[i] & exponent_mask;
    exponent = exponent >> 7;

    // 1st check for zero.  This is a special case because
    // it cannot actually be represented because there is an
    // assumed 1 before the mantissa.
    if ((exponent == 0) && (mantissa == 0)) {
      // don't need to do anything because 0 happens to be the same
      // in both vax and IEEE
      continue;
    }

    // 2nd check for exponents which when adjusted would
    // be out of range on IEEE and hard code these to 0.
    if (exponent <= 3) {
      // exponent <= 2 adjustment would wrap.
      // exponent == 3 used by IEEE for very near zero and 
      //               has no correspondance in VAX
      u4[i] = 0;
      continue;
    }

    exponent = exponent - 2;

    // put the exponent back
    u4[i] = u4[i] & ~exponent_mask;
    u4[i] = u4[i] | (exponent << 7);
  }

#ifdef M2_IEEE_
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#elif defined(M2_EEEI_)
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
  Swap4(buffer, n);
#elif M2_VAX_
  Swap4(buffer,n);
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#endif
#endif	// CNVX_

} // f_vax2ieee
 


static void f_vax2eeei (real_4 *buffer, Size n)
{
//  cerr << "f_vax2eeei" << endl;
  f_vax2ieee(buffer, n);
  f_ieee2eeei(buffer, n);
} // f_vax2eeei



// d_vax2eeei converts from D_FLOAT VAX double precision
// to little endian IEEE.

static void d_vax2eeei (real_8 *buffer, Size n)
{
//  cerr << "d_vax2eeei n is " << n << endl;
  int i;
  int_u8 *u8 =(int_u8*)buffer;

  int nnn = n;
  // First, word swap to match local machine.
  
#ifdef M2_IEEE_
  Swap2(buffer, n * 8);
#elif defined(M2_EEEI_)
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < nnn * 4; i = i + 4) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+3];
    bufferAsU2[i+3] = tempU2;
    tempU2 = bufferAsU2[i+1];
    bufferAsU2[i+1] = bufferAsU2[i+2];
    bufferAsU2[i+2] = tempU2;
  }
#elif M2_VAX_
#endif


  //RTS static const int_u8 vax_sign_mask     = 0x8000000000000000LL;
  static const int_u8 vax_exponent_mask = 0x7f80000000000000LL;
  static const int_u8 vax_mantissa_mask = 0x007fffffffffffffLL;

  //RTS static const int_u8 ieee_sign_mask     = 0x8000000000000000LL;
  static const int_u8 ieee_exponent_mask = 0x7ff0000000000000LL;
  static const int_u8 ieee_mantissa_mask = 0x000fffffffffffffLL;

//  int_u8 sign;
  int_u8 exponent;
  int_u8 mantissa;

  for (i = 0; i < nnn; i++) {

    // just to show where the sign is 
    //    sign = u8[i] & sign_mask;
    mantissa = u8[i] & vax_mantissa_mask;
    exponent = u8[i] & vax_exponent_mask;

    // 1st check for zero.  This is a special case because
    // it cannot actually be represented because there is an
    // assumed 1 before the mantissa.
    if ((exponent == 0) && (mantissa == 0)) {
      // don't need to do anything because 0 happens to be the same
      // in both vax and IEEE
      // kill the sign
      u8[i] = 0;
      continue; 
    }

    mantissa = mantissa >> 3;
    exponent = exponent >> 55;
    exponent = exponent - 127;
    exponent = exponent - 2;
    exponent = exponent + 1023;
    exponent = exponent << 52;

    // put the mantissa back
    u8[i] = u8[i] & ~ieee_mantissa_mask;
    u8[i] = u8[i] | mantissa;

    // put the exponent back
    u8[i] = u8[i] & ~ieee_exponent_mask;
    u8[i] = u8[i] | exponent;
  }

#ifdef M2_IEEE_
#elif defined(M2_EEEI_)
#elif M2_VAX_
  Swap8(buffer,n);
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * n; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#endif

} // d_vax2eeei



// d_vax2ieee converts from D_FLOAT VAX double precision
// to big endian IEEE.

static void d_vax2ieee (real_8 *buffer, Size n)
{
//  cerr << "d_vax2ieee" << endl;
#if CNVX_
  Size nbytes = n * sizeof(double);
  cvt$vaxdToIEEE(fv, fv, &nbytes);
#else
  int i;

  int nnn = n;
  // First, word swap to match local machine.

#ifdef M2_IEEE_
  Swap2(buffer, nnn * 4);
#elif defined(M2_EEEI_)
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 4) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+3];
    bufferAsU2[i+3] = tempU2;
    tempU2 = bufferAsU2[i+1];
    bufferAsU2[i+1] = bufferAsU2[i+2];
    bufferAsU2[i+2] = tempU2;
  }
#elif M2_VAX_
#endif


  int_u8 *u8 =(int_u8*)buffer;

  //RTS static const int_u8 vax_sign_mask     = 0x8000000000000000LL;
  static const int_u8 vax_exponent_mask = 0x7f80000000000000LL;
  static const int_u8 vax_mantissa_mask = 0x007fffffffffffffLL;

  //RTS static const int_u8 ieee_sign_mask     = 0x8000000000000000LL;
  static const int_u8 ieee_exponent_mask = 0x7ff0000000000000LL;
  static const int_u8 ieee_mantissa_mask = 0x000fffffffffffffLL;

//  int_u8 sign;
  int_u8 exponent;
  int_u8 mantissa;

  for (i = 0; i < nnn; i++) {
    // just to show where the sign is
    //    sign = u8[i] & sign_mask;
    mantissa = u8[i] & vax_mantissa_mask;
    exponent = u8[i] & vax_exponent_mask;

    // 1st check for zero.  This is a special case because
    // it cannot actually be represented because there is an
    // assumed 1 before the mantissa.
    if ((exponent == 0) && (mantissa == 0)) {
      // don't need to do anything because 0 happens to be the same
      // in both vax and IEEE
      u8[i] = 0;
      continue;
    }


    mantissa = mantissa >> 3;
    exponent = exponent >> 55;
    exponent = exponent - 127;
    exponent = exponent - 2;
    exponent = exponent + 1023;
    exponent = exponent << 52;

    // put the mantissa back
    u8[i] = u8[i] & ~ieee_mantissa_mask;
    u8[i] = u8[i] | mantissa;

    // put the exponent back
    u8[i] = u8[i] & ~ieee_exponent_mask;
    u8[i] = u8[i] | exponent;
  }

#ifdef M2_IEEE_
#elif defined(M2_EEEI_)
  bufferAsU2 = (int_u2*)buffer;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#elif M2_VAX_
  Swap8(buffer,n);
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#endif

#endif	// CNVX_

} // d_vax2ieee 
 



// ///////////////////////////////////////////// IEEE/VAX Conversions

// d_ieee2vax converts from big endian IEEE to D_FLOAT VAX double precision.

static void f_ieee2vax (real_4* buffer, Size n)
{
  int nnn = n;
//  cerr << "f_ieee2vax" << endl;
#if CNVX_
  Size nbytes = n * sizeof(float);
  cvt$IEEEtoVaxf(fv, fv, &nbytes);
#else
  int i;
  // 1st, word swap to match local machine.
#ifdef M2_IEEE_
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#elif defined(M2_EEEI_)
   int_u1* bufferAsU1 = (int_u1*)buffer;
   int_u1 tempU1;
   for (i = 0; i < 4 * nnn; i = i + 2) {
     tempU1 = bufferAsU1[i];
     bufferAsU1[i] = bufferAsU1[i+1];
     bufferAsU1[i+1] = tempU1;
   }
#elif M2_VAX_
    int_u1* bufferAsU1 = (int_u1*)buffer;
    int_u1 tempU1;
    for (i = 0; i < 4 * nnn; i = i + 2) {
      tempU1 = bufferAsU1[i];
      bufferAsU1[i] = bufferAsU1[i+1];
      bufferAsU1[i+1] = tempU1;
    }
#endif

  int_u4 *u4 =(int_u4*)buffer;

//  int sign;
  int exponent;
  int mantissa;

  for (i = 0; i < nnn; i++) {
    //RTS int_u4 sign_mask = 1 << 15;
    int_u4 exponent_mask = (0xff << 7);

    // just to show where the sign is
    //    sign = u4[i] & SIGN_MASK;
    //    sign = sign >> 15;
    mantissa = u4[i] & 0x7f;
    mantissa = mantissa << 16;
    mantissa = mantissa | (u4[i] & 0xffff0000) >> 16;
    exponent = u4[i] & exponent_mask;
    exponent = exponent >> 7;

    // 1st check for zero.  This is a special case because
    // it cannot actually be represented because there is an
    // assumed 1 before the mantissa.
    if ((exponent == 0) && (mantissa == 0)) {
      // don't need to do anything because 0 happens to be the same
      // in both vax and IEEE
      u4[i] = 0; // kill the sign bit
      continue;
    }

    // 2nd check for exponents which when adjusted would
    // be out of range on IEEE and hard code these to 0.
    if (exponent >= 254) {
      // exponent >= 254 adjustment would wrap; set value to max
      u4[i] = u4[i] | exponent_mask; // exponent = 255
      u4[i] = u4[i] | 0xffff007f;    // matissa = all 1
      continue;
    }

    // 3rd. check for unrepresentable values.
    if (exponent == 1) {
       // exponent == 3 (1 IEEE) is  used by IEEE for very near zero and 
       //               has no correspondance in VAX; set value to 0
      u4[i] = 0;
      continue;
    }

    exponent = exponent + 2;

    // put the exponent back
    u4[i] = u4[i] & ~exponent_mask;
    u4[i] = u4[i] | (exponent << 7);
  }
#ifdef M2_IEEE_
  Swap4(buffer,n);
#elif defined(M2_EEEI_)
#elif M2_VAX_
#endif
#endif
} // f_ieee2vax 
 


static void d_ieee2vax (double *buffer, Size n)
{
  int nnn = n;
//  cerr << "d_ieee2vax n is " << n << endl;
#if CNVX_
  Size nbytes = nnn * sizeof(double);
  cvt$IEEEtoVaxd(buffer, buffer, &nbytes);
#else
  int i;

// word swap so that native machine may look at it as 
// int_u8.
#ifdef M2_IEEE_
  // already in the correct order
#elif defined(M2_EEEI_)
  Swap8(buffer,n);
#elif M2_VAX_
#endif

  int_u8 *u8 =(int_u8*)buffer;

  //RTS static const int_u8 vax_sign_mask     = 0x8000000000000000LL;
  static const int_u8 vax_exponent_mask = 0x7f80000000000000LL;
  static const int_u8 vax_mantissa_mask = 0x007fffffffffffffLL;

  //RTS static const int_u8 ieee_sign_mask     = 0x8000000000000000LL;
  static const int_u8 ieee_exponent_mask = 0x7ff0000000000000LL;
  static const int_u8 ieee_mantissa_mask = 0x000fffffffffffffLL;

//  int_u8 sign;
  int_8 exponent;
  int_u8 mantissa;

  for (i = 0; i < nnn; i++) {
//    cerr << "here ";
    // just to show where the sign is
    //    sign = u8[i] & sign_mask;
    mantissa = u8[i] & ieee_mantissa_mask;
    exponent = u8[i] & ieee_exponent_mask;

    // 1st check for zero.  This is a special case because
    // it cannot actually be represented because there is an
    // assumed 1 before the mantissa.
    if ((exponent == 0) && (mantissa == 0)) {
      // kill the sign bit
      u8[i] = 0;
      continue;
    }

    exponent = exponent >> 52;
    exponent = exponent - 1023;

    // the number is too big or too negative to be represented
    // in vax d format so set it to the max value.
    if (exponent >= 127) {
      u8[i] = u8[i] | 0x7f80000000000000LL | 0x007fffffffffffffLL;
      continue;
    }

    // the number is too small ( close to 0 ) so set it to 0.
    if (exponent <= -129) {
      u8[i] = 0;
      continue;
    }
    
    exponent = exponent + 2;
    exponent = exponent + 127;
    exponent = exponent << 55;

    mantissa = mantissa << 3;
    // put the mantissa back
    u8[i] = u8[i] & ~vax_mantissa_mask;
    u8[i] = u8[i] | mantissa;

    
    // put the exponent back
    u8[i] = u8[i] & ~vax_exponent_mask;
    u8[i] = u8[i] | exponent;
  }

// convert to VAX word order
#ifdef M2_IEEE_
  Swap2(buffer, n * 4);
#elif defined(M2_EEEI_)
   int_u2* bufferAsU2 = (int_u2*)buffer;
   int_u2 tempU2;
   for (i = 0; i < 4 * nnn; i = i + 4) {
     tempU2 = bufferAsU2[i];
     bufferAsU2[i] = bufferAsU2[i+3];
     bufferAsU2[i+3] = tempU2;
     tempU2 = bufferAsU2[i+1];
     bufferAsU2[i+1] = bufferAsU2[i+2];
     bufferAsU2[i+2] = tempU2;
   }
#elif M2_VAX_
  Swap8(buffer,n);
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < 2 * nnn; i = i + 2) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+1];
    bufferAsU2[i+1] = tempU2;
  }
#endif
#endif
} // d_ieee2vax 
 


static void f_eeei2vax (real_4 *buffer, Size n)
{
//  cerr << "f_eeei2vax" << endl;
  f_eeei2ieee(buffer, n);
  f_ieee2vax(buffer, n);
} // f_eeei2vax



// d_eeei2vax converts from little endian IEEE to D_FLOAT VAX double
// precision.

static void d_eeei2vax (real_8 *buffer, Size n)
{
  int nnn = n;
//  cerr << "d_eeei2vax" << endl;
  int i;
  int_u8 *u8 =(int_u8*)buffer;

// word swap so that native machine may look at it as 
// int_u8.
#ifdef M2_IEEE_
  // already in the correct order
#elif defined(M2_EEEI_)
#elif M2_VAX_
#endif


  //RTS static const int_u8 vax_sign_mask     = 0x8000000000000000LL;
  static const int_u8 vax_exponent_mask = 0x7f80000000000000LL;
  static const int_u8 vax_mantissa_mask = 0x007fffffffffffffLL;

  //RTS static const int_u8 ieee_sign_mask     = 0x8000000000000000LL;
  static const int_u8 ieee_exponent_mask = 0x7ff0000000000000LL;
  static const int_u8 ieee_mantissa_mask = 0x000fffffffffffffLL;

//  int_u8 sign;
  int_8 exponent;
  int_u8 mantissa;

  for (i = 0; i < nnn; i++) {
    // just to show where the sign is
    //    sign = u8[i] & sign_mask;
    mantissa = u8[i] & ieee_mantissa_mask;
    exponent = u8[i] & ieee_exponent_mask;

    // 1st check for zero.  This is a special case because
    // it cannot actually be represented because there is an
    // assumed 1 before the mantissa.
    if ((exponent == 0) && (mantissa == 0)) {
      // kill the sign bit
      u8[i] = 0;
      continue;
    }

    exponent = exponent >> 52;
    exponent = exponent - 1023;

    // the number is too big or too negative to be represented
    // in vax d format so set it to the max value.
    if (exponent >= 127) {
      exponent = 126;
      mantissa = ieee_mantissa_mask;
    }

    // the number is too small ( close to 0 ) so set it to 0.
    if (exponent <= -129) {
      u8[i] = 0;
      continue;
    }
    
    exponent = exponent + 2;
    exponent = exponent + 127;
    exponent = exponent << 55;

    mantissa = mantissa << 3;
    // put the mantissa back
    u8[i] = u8[i] & ~vax_mantissa_mask;
    u8[i] = u8[i] | mantissa;

    
    // put the exponent back
    u8[i] = u8[i] & ~vax_exponent_mask;
    u8[i] = u8[i] | exponent;
  }

// convert to VAX word order
#ifdef M2_IEEE_
  Swap2(buffer, nnn * 4);
#elif defined(M2_EEEI_)
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < nnn * 4; i = i + 4) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+3];
    bufferAsU2[i+3] = tempU2;
    tempU2 = bufferAsU2[i+1];
    bufferAsU2[i+1] = bufferAsU2[i+2];
    bufferAsU2[i+2] = tempU2;
  }
#elif M2_VAX_
  int_u2* bufferAsU2 = (int_u2*)buffer;
  int_u2 tempU2;
  for (i = 0; i < nnn * 4; i = i + 4) {
    tempU2 = bufferAsU2[i];
    bufferAsU2[i] = bufferAsU2[i+3];
    bufferAsU2[i+3] = tempU2;
    tempU2 = bufferAsU2[i+1];
    bufferAsU2[i+1] = bufferAsU2[i+2];
    bufferAsU2[i+2] = tempU2;
  }
#endif
} // d_eeei2vax



class UnsupportedMachineRepConvertEx : public runtime_error {
public:
  UnsupportedMachineRepConvertEx(MachineRep_e l, MachineRep_e r) : 
    runtime_error("can't convert from machine rep"+Stringize(l)+" to "+Stringize(r)) { }
}; 
// ///////////////////////////////////////////// CRAY/EEEI Conversions


static void f_eeei2cray (real_4 *, Size sz)
{
  if (sz>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_EEEI, MachineRep_CRAY);

} // f_eeei2cray



static void f_cray2eeei (real_4 *, Size sz)
{
  if (sz>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_CRAY, MachineRep_EEEI);

} // f_cray2eeei



static void d_eeei2cray (real_8 *, Size sz)
{
  if (sz>0) 
    throw UnsupportedMachineRepConvertEx(MachineRep_EEEI, MachineRep_CRAY);

} // d_eeei2cray



static void d_cray2eeei (real_8 *, Size sz)
{
  if (sz>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_CRAY, MachineRep_EEEI);

} // d_cray2eeei


// ///////////////////////////////////////////// IEEE/CRAY Conversions

static void f_ieee2cray (float *fv, Size n)
{
#if CNVX_
  Size nbytes = n * sizeof(float);
  cvt$ieee2ibm(fv, fv, &nbytes);
#else
  float* disable_warn = fv; if(disable_warn) {};
  if (n>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_IEEE, MachineRep_CRAY);
#endif

} // f_ieee2cray 


 
static void f_cray2ieee (float *fv, Size n)
{
#if CNVX_
  Size nbytes;

  nbytes = n * sizeof(float);
  cvt$ibm2ieee(fv, fv, &nbytes);
#else
  float* disable_warn = fv; if (disable_warn) {};
  if (n>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_CRAY, MachineRep_IEEE);
#endif

} // f_cray2ieee 
 


static void d_ieee2cray (double *dv, Size n)
{
#if CNVX_
  Size nbytes;

  nbytes = n * sizeof(double);
  cvt$IEEEtoCrayf (dv, dv, &nbytes);
#else
  double* disable_warning_dv = dv; if (disable_warning_dv) {};
  if (n>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_IEEE, MachineRep_CRAY);
#endif

} // d_ieee2cray 
 


static void d_cray2ieee (double *dv, Size n)
{
#if CNVX_
  Size nbytes;

  nbytes = n * sizeof(double);
  cvt$crayfToIEEE(dv, dv, &nbytes);
#else
  double* disable_warning_dv = dv; if (disable_warning_dv) {};
  if (n>0)
    throw UnsupportedMachineRepConvertEx(MachineRep_CRAY, MachineRep_IEEE);
#endif

} // d_cray2ieee 




// ///////////////////////////////////////////// TI 320 DSP Routines

//static 
void f_ti2eeei (int_4 *fv, Size n)
{
  int_4	exponent;
  int_4	mantissa;
  int_4	float_320;

  union	{
    long ieee_hex;
    float ieee_float;
  } converted;

  for (Index i = 0; i < n; i++) {
    float_320 = fv[i];
    exponent = float_320 >> 24;		// isolate the 320 exponent
    converted.ieee_float = 0.0;		// zero the return value
    if( exponent != -128 ) {		// if exponent = -128, all done
      exponent += 127;			// convert exponent to offset form
      mantissa = float_320 & 0xffffff;	// isolate the 320 mantissa
      if( mantissa & 0x800000 ) {	// check for negative mantissa
        converted.ieee_hex = 0x80000000;// set sign bit
        mantissa &= 0x7fffff;		// strip off sign bit
        mantissa |= 0xff000000;		// sign extend mantissa
        mantissa = ~mantissa + 1;	// take two's complement
        mantissa &= 0x7fffff;		// strip off 'hidden' bit
        if( !mantissa ) exponent++;	// increment exponent for -1
      }
      converted.ieee_hex |= ( ( exponent << 23 ) | mantissa );
    }
    fv[i] = converted.ieee_hex;
  }
} // f_ti2eeei 



//static 
void f_eeei2ti (int_4 *fv, Size n)
{
  int_4	exponent;
  int_4	mantissa;
  int_4	ieee_hex;
  int_4	float_320;

  for (Index i = 0; i < n; i++) {
    ieee_hex = fv[i];
    exponent = (ieee_hex >> 23) & 0xff;	// isolate the ieee exponent
    mantissa = ieee_hex & 0x7fffff;	// isolate the ieee mantissa
    exponent -= 127;			// convert exponent from offset form
    if (ieee_hex & 0x80000000) {
      if( !mantissa ) exponent--;	// decrement exponent for -1
      mantissa = ~(mantissa-1);		// take two's complement
      mantissa &= 0x00ffffff;		// clear high bits
      mantissa |= 0x00800000;		// add sign bit
    } 
    float_320 = (  ( exponent << 24 ) | mantissa );
    fv[i] = float_320;
  }
} // f_eeei2ti 




// ///////////////////////////////////////////// Buffer Conversion Routines

inline static Size true_byte_length (Numeric_e format, Size elements)
{
  switch (format) {
    case BIT:
      {
	Size byte_length = elements / 8;
	if ((byte_length * 8) < elements)
	  return byte_length + 1;
	else
	  return byte_length;
      }
      M2BOGUS_BREAK;

    case CX_BIT:
      {
	Size byte_length = elements / 4;
	if ((byte_length * 4) < elements)
	  return byte_length + 1;
	else
	  return byte_length;
      }
      M2BOGUS_BREAK;

    default:
      return elements * ByteLength(format);
  }
}					// true_byte_length 



void ConvertBufferRep (MachineRep_e in_rep, MachineRep_e out_rep,
		       const void* in_buf, void* out_buf,
		       Numeric_e format, int_4 elements)
{
  if (in_buf != out_buf) {
    memmove(out_buf, in_buf, true_byte_length(format, elements));
  }

  ConvertBufferRepInPlace(in_rep, out_rep, out_buf, format, elements);

}					// ConvertBufferRep


void ConvertBufferRepInPlace (MachineRep_e in_rep, MachineRep_e out_rep,
			      void* buf, Numeric_e format, int_4 elements)
{
  if (in_rep == out_rep)
    return;

  switch (format) {
    case BIT: case CX_BIT:
    case BYTE: case CX_BYTE:
    case UBYTE: case CX_UBYTE:
      return;
      M2BOGUS_BREAK;
    default: break; // Just fall through I guess?  
  }

  if (isComplex(format)) {
    elements *= 2;
    format = toReal(format);
  }

  switch (in_rep) {

    case MachineRep_IEEE:		// From IEEE Representation...
      switch (out_rep) {

	case MachineRep_EEEI:		// IEEE to EEEI
	break;

	case MachineRep_VAX:		// IEEE to VAX
	  switch (format) {
	    case DOUBLE:
	      d_ieee2vax(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_ieee2vax(reinterpret_cast(real_4*, buf), elements);
	      return;

	    default: break; // Just fall through I guess?  
	  }
	  break;

	case MachineRep_CRAY:		// IEEE to CRAY
	  switch (format) {
	    case DOUBLE:
	      d_ieee2cray(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_ieee2cray(reinterpret_cast(real_4*, buf), elements);
	      return;
	    default: break; // Just fall through I guess?  
	  }
	  break;

	default:
	  throw UnsupportedMachineRepConvertEx(in_rep, out_rep);
      }
      break;

    case MachineRep_EEEI:		// From EEEI Representation...
      switch (out_rep) {

	case MachineRep_IEEE:		// EEEI to IEEE
	  break;

	case MachineRep_VAX:		// EEEI to VAX
	  switch (format) {
	    case DOUBLE:
	      d_eeei2vax(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_eeei2vax(reinterpret_cast(real_4*, buf), elements);
	      return;
	    default: break; // Just fall through I guess?  
	  }
	  break;

	case MachineRep_CRAY:		// EEEI to CRAY
	  switch (format) {
	    case DOUBLE:
	      d_eeei2cray(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_eeei2cray(reinterpret_cast(real_4*, buf), elements);
	      return;
	    default: break; // Just fall through I guess?  
	  }
	  break;

	default:
	  throw UnsupportedMachineRepConvertEx(in_rep, out_rep);
      }
      break;

    case MachineRep_VAX:		// From VAX Representation...
      switch (out_rep) {
	case MachineRep_IEEE:		// VAX to IEEE
	  switch (format) {
	    case DOUBLE:
	      d_vax2ieee(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_vax2ieee(reinterpret_cast(real_4*, buf), elements);
	      return;
	    default: break; // Just fall through I guess?  
	  }
	  break;

	case MachineRep_EEEI:		// VAX to EEEI
	  switch (format) {
	    case DOUBLE:
	      d_vax2eeei(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_vax2eeei(reinterpret_cast(real_4*, buf), elements);
	      return;
	    default: break; // Just fall through I guess?  
	  }
	  break;

	default:
	  throw UnsupportedMachineRepConvertEx(in_rep, out_rep);
      }
      break;


    case MachineRep_CRAY:		// From VAX Representation...
      switch (out_rep) {
	case MachineRep_IEEE:		// CRAY to IEEE
	  switch (format) {
	    case DOUBLE:
	      d_cray2ieee(reinterpret_cast(real_8*, buf), elements);
	      break;

	    case FLOAT:
	      f_cray2ieee(reinterpret_cast(real_4*, buf), elements);
	      break;
            default: break; // Just fall through I guess?  
	  }
	  break;



	case MachineRep_EEEI:		// CRAY to EEEI
	  switch (format) {
	    case DOUBLE:
	      d_cray2eeei(reinterpret_cast(real_8*, buf), elements);
	      return;

	    case FLOAT:
	      f_cray2eeei(reinterpret_cast(real_4*, buf), elements);
	      return;
	    default: break; // Just fall through I guess?  
	  }
	  break;


	default:
	  throw UnsupportedMachineRepConvertEx(in_rep, out_rep);
      }
      break;

    default:
      throw UnsupportedMachineRepConvertEx(in_rep, out_rep);

  } // switch (in_rep)

  if (IsBigEndian(in_rep) != IsBigEndian(out_rep)) {
    switch (ByteLength(format)) {
    case 8:
      Swap8(buf, elements);
      break;
    case 4:
      Swap4(buf, elements);
      break;
    case 2:
      Swap2(buf, elements);
      break;
    case 1:
      break;
    }
  }

}					// ConvertBufferRepInPlace


PTOOLS_END_NAMESPACE

