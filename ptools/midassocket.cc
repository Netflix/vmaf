
#include "midassocket.h"
#include "socketerror.h"
#if defined(_MSC_VER) || defined(__MINGW32__)
#else
#include <netdb.h>   // for gethostbyname ... this causes conflicts
                     // on tru64 with X-Midas, so it has been moved to a .cc
#include <sys/time.h>
#include <unistd.h>
#endif

#include <math.h>
#include <fcntl.h>
#include <errno.h>
#include "m2convertrep.h"
#include "m2streamdataenc.h"

PTOOLS_BEGIN_NAMESPACE

// This is incredible ugly code from M2k:  We preserve it "as-is"
// to get the most backward compatible piece of it.
int MidasSocket_::handleReadingM2kHdr_ (int fd, char* rep, 
					MachineRep_e& endian)
{
  StreamDataEncoding sde;
  readContinueStreamDataEncoding_(fd, sde, rep);
  int correction = 0;  // The msglen is still the same, the extra 4 bytes
                       // were for further header
  endian = sde.machineRep();
  return correction;
}


// STRAIGHT from M2K;  don't touch unless you have to!!

// This is a helper routine used by "guess":
// This is all "cut-and-paste" code from BStream::readStreamDataEncoding.
// Because it is baseline code, we can't really change it, so yes,
// this is brittle and probably should be a method of the BStream
// if we do put this functionality in the baseline.
void MidasSocket_::readContinueStreamDataEncoding_ (int fd,
						   StreamDataEncoding& sde,
						   char* rep)
{
  // Prior to M2k version 3.0.6.0, the initial four bytes were the
  // string format of the machine rep, and that's all there was.
  // Determine whether we are using an extended-format encoding
  // message, in which case there is more stuff that we have to read.
  // If not, the test will at least record the correct machine
  // representation in the sde.
  int_u1* buf;
  Size len;
  if (sde.isExtendedFormat(rep, 4, buf, len)) {
    // We are using an extended format.  The length of the message
    // may change, but there's a prefix that is known to be the
    // same, and the SDE told us how long it is and where we
    // should put the rest of it.  Do so.
    // RTS: bin.readExact(buf, len);
    readExact_(fd, (char*)buf, len);

    // Now ask the SDE to look at the prefix, and determine how much
    // more information is needed to complete the SDE message.  Read
    // that in, then have the SDE initialize itself from the entire
    // message.
    sde.extendMessageFromPrefix(buf, len);
    // RTS: bin.readExact(buf, len);
    readExact_(fd, (char*)buf, len);

    sde.setFromMessage();
  } else {
    // Not using extended format; presumably, the other side is a version
    // of m2k that is pretty old.  Set the default version number to
    // Original.
    sde.defaultSerialFormatLatest(false);
  }
  return;
}




static const int_u4 Strings__SFV_Latest = 1;  // from m2strings.h    
static const int_u4 M2Time__SFV_Latest = 2;   // from m2time.h and m2memstream
static const int_u4 EventData__SFV_Latest = 3;// from m2eventdata.cc
static const int_u4 TimePacket__SFV_Latest = 3; // from m2timepacket.cc
static const int_u4 TimeStamp__SFV_Latest = 4;  // from same
static const int_u4 IndexedTimeStamp__SFV_Latest = 2; // same


void MidasSocket_::handleWritingM2kHdr_(int fd) 
{
  StreamDataEncoding sde;

  // This block of code comes from OpalValue (m2opalvalue.cc)
  { //  OpalValue::setStreamDataEncodingVersions(sde);

    // Here are listed encoding versions for all serializable items that
    // have changed their serialization format over time.  Indicate in
    // the encoding structure that we will be serializing in the most
    // recent format.
    sde.serialFormatVersion(StreamDataEncoding::SFI_Strings_ie,
			    Strings__SFV_Latest);
    sde.serialFormatVersion(StreamDataEncoding::SFI_M2Time_ie,
			    M2Time__SFV_Latest);
    sde.serialFormatVersion(StreamDataEncoding::SFI_EventData_ie,
			    EventData__SFV_Latest);
    sde.serialFormatVersion(StreamDataEncoding::SFI_TimePacket_ie,
			    TimePacket__SFV_Latest);
    sde.serialFormatVersion(StreamDataEncoding::SFI_TimeStamp_ie,
			    TimeStamp__SFV_Latest);
    sde.serialFormatVersion(StreamDataEncoding::SFI_IndexedTimeStamp_ie,
			    IndexedTimeStamp__SFV_Latest);
  }

  // This block of code comes BStream::writeStreamDataEncoding (m2bstream.cc)
  { // bout.writeStreamDataEncoding(sde);

    // Somebody else has already set all the serialization format versions
    // and other material.  Make the SDE write itself into a message buffer,
    // and tell us where that buffer is and how long it is; then write the
    // buffer out over the stream.
    
    int_u1* mbuf;
    Size mlen;
    
    sde.setupMessageBuffer(mbuf, mlen);
    writeExact_(fd, (char*)mbuf, mlen);
  }
}


PTOOLS_END_NAMESPACE
