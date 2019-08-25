//
// ///// Authors: Peter Bigot
//

// ///////////////////////////////////////////// Include Files

#include "m2streamdataenc.h"
#include <stdlib.h>			// for malloc, free
#include <string.h>			// for memcpy

PTOOLS_BEGIN_NAMESPACE

// ///////////////////////////////////////////////////// Constant Values

// How many low-value tags do we expect?  Keep this in constant-time access
// memory.
#define SDEC_TagTableSize 64

// Version representing the generic "original" encoding.
const int_u4 StreamDataEncoding::SFV_Original = 0;

// Version representing the generic "latest" encoding.  Must be translated on a
// per-class basis to the real latest version number.

const int_u4 StreamDataEncoding::SFV_Latest = 0xFFFFFFFF;

// How long is the marker sequence?  It's four bytes, and normally we'd use
// sizeof(marker), but DEC has certain instances (like when the marker is
// passed in a method argument) when it thinks sizeof(char[4]) is 8.  So
// we'll use this macro instead of "4" or "sizeof(rep)" everywhere.
#define SDEC_MarkerLength (4)

// The marker tag that we use to denote extended format serial encoding messages.
static const char SDEMessageMarker_[SDEC_MarkerLength] = { 'M', '2', 'B', 'D' };


// ///////////////////////////////////////////////////// Type Declarations

// Generic prefix that all formats of SDEMessage must conform to.
struct SDEMessage_0_ {
    // BEGIN IMMUTABLE REGION (all revisions of message must begin with these 8 bytes)
    char marker_[SDEC_MarkerLength];// M2BD  == Midas 2k Binary Data
    int_u1 reserved_;         // Reserved, must be zero
    int_u1 versionNumber_;    // ??
    int_u2 length_;           // Length of message, in bytes, including this prefix, network byte order
    // END IMMUTABLE REGION
};


// Pair of type tag and serialization version, used in format 1 of SDEMessage.
struct TagVersionPair_1_ {
    int_u4 tag;
    int_u4 version;
};

// Version 1 of SDEMessage: includes representation, source M2k version name, and
// a set of tag/version pairs represented in TagVersionPair_1_ structures.
struct SDEMessage_1_ {
    // BEGIN IMMUTABLE REGION (all revisions of message must begin with these 8 bytes)
    char marker_[SDEC_MarkerLength];// M2BD  == Midas 2k Binary Data
    int_u1 reserved_;         // 0
    int_u1 versionNumber_;    // 1
    int_u2 length_;           // Length of message, in bytes, including this prefix, network byte order
    // END IMMUTABLE REGION
    char machineRep_[SDEC_MarkerLength];// Standard representation
    int_u4 sourceVersionLength_; // Length in bytes of string
    int_u4 numberOfSFVs_;	// Number of tag/version pairs
    char body_[1];
    // Dynamically-sized portions of message (overlaps body_)
    // TagVersionPair_1_ tagver_[numberOfSFVs_];  /* starts at body_ */
    // char[] sourceVersion_; /* at tagver_ + numberOfSFVs_*sizeof(tagver_[0]) */
    // Data ends at sourceVersion_ + sourceVersionLength*sizeof(sourceVersion_[0])
};


// ///////////////////////////////////////////////////// Method Definitions


StreamDataEncoding::StreamDataEncoding () :
  useMachineRepOnly_(false),
  defaultSFV_(StreamDataEncoding::SFV_Latest),
  versionNumber_(0),
  sourceVersion_(MIDAS_VERSION),
  machineRep_(NativeMachineRep()),
  messageBuffer_(0),
  messageLength_(0)
{
  sfValidVersion_.expandTo(SDEC_TagTableSize);
  sfVersion_.expandTo(SDEC_TagTableSize);
  reset();
}



StreamDataEncoding::~StreamDataEncoding ()
{
  freeMessageBuffer_();
  messageLength_ = 0;
}



void StreamDataEncoding::reset ()
{
  // Reset the machine representation and clear all information about
  // serialization versions.  DO NOT free the message buffer; we'll do that
  // later when we have to.
  machineRep_ = MachineRep_UNDEFINED;
  useMachineRepOnly_ = false;
  bool* wp = sfValidVersion_.data();
  memset(wp, 0, sfValidVersion_.length()*sizeof(*wp));
  sfExtendedTag_.clear();
  sfExtendedVersion_.clear();
  sourceVersion_ = "Unknown";
  return;
}



// Pack our data into a version-1 SDEMessage.

void StreamDataEncoding::packMessage_1_ ()
{
  Size npairs = 0;

  // How many tag/version pairs do we need to transfer?
  Index i;
  for (i = 0; i < sfValidVersion_.length(); i++) {
    if (sfValidVersion_[i]) {
      ++npairs;
    }
  }
  npairs += sfExtendedTag_.length();

  // Compute offsets of the dynamically-sized portions of the message,
  // and the length of the message
  int ofs_tagver = offsetof(SDEMessage_1_, body_);
  TagVersionPair_1_* tvp;
  int ofs_sourceVersion = ofs_tagver + npairs * sizeof(*tvp);
  messageLength_ = ofs_sourceVersion + sizeof(MIDAS_VERSION);

  // Allocate a new buffer for the message, and get an SDE-style pointer
  // into it.  Zero out the buffer to avoid UMR errors from Purify.
  freeMessageBuffer_();
  messageBuffer_ = new int_u1[messageLength_];
  memset(messageBuffer_, 0, messageLength_);
  SDEMessage_1_* msmp = reinterpret_cast(SDEMessage_1_*, messageBuffer_);

  // Copy over the marker tag, and set up the reset of the standard
  // prefix.
  memcpy(msmp->marker_, SDEMessageMarker_, sizeof(msmp->marker_));
  msmp->reserved_ = 0;
  msmp->versionNumber_ = 1;
  msmp->length_ = messageLength_;

  // Convert the length into network byte order.
  MachineRep_e mrep = NativeMachineRep();
  ConvertBufferRepInPlace(mrep, MachineRep_NETWORK, &msmp->length_, UINTEGER, 1);

  // Encode the machine representation as a 4-byte string
  string s = EncodeMachineRep(mrep);
  int sl = s.length();
  for (i = 0; i < int(sizeof(msmp->machineRep_)); i++) {
    msmp->machineRep_[i] = (i < Index(sl)) ? s[i] : ' ';
  }

  msmp->numberOfSFVs_ = npairs;
  
  tvp = reinterpret_cast(TagVersionPair_1_*, ofs_tagver + reinterpret_cast(int_u1*, msmp));
  for (i = 0; i < sfValidVersion_.length(); i++) {
    if (sfValidVersion_[i]) {
      tvp->tag = i;
      tvp->version = sfVersion_[i];
      ++tvp;
    }
  }
  for (i = 0; i < sfExtendedTag_.length(); i++) {
    tvp->tag = sfExtendedTag_[i];
    tvp->version = sfExtendedVersion_[i];
    ++tvp;
  }

  char* cp = reinterpret_cast(char*, ofs_sourceVersion + reinterpret_cast(int_u1*, msmp));
  string mver(MIDAS_VERSION);
  msmp->sourceVersionLength_ = mver.length();
  memcpy(cp, mver.data(), msmp->sourceVersionLength_);

  return;
}



// Unpack a version-1 format SDE message.  The SDE class members should
// have already been reset.

void StreamDataEncoding::unpackMessage_1_ ()
{
  SDEMessage_1_* msmp = reinterpret_cast(SDEMessage_1_*, messageBuffer_);

  // Pull out the machine rep.
  machineRep_ = DecodeMachineRep(string(msmp->machineRep_, sizeof(msmp->machineRep_)));

  // Convert count values into native machine representation.
  Size npairs = NativeMachineRep(machineRep_, msmp->numberOfSFVs_);
  Size svl = NativeMachineRep(machineRep_, msmp->sourceVersionLength_);

  // Get the offsets of the dynamically-sized portions of the message
  int ofs_tagver = offsetof(SDEMessage_1_, body_);
  TagVersionPair_1_* tvp;
  int ofs_sourceVersion = ofs_tagver + npairs * sizeof(*tvp);
  
  // Extract the tag/version pairs and install them in the tables.
  tvp = reinterpret_cast(TagVersionPair_1_*, ofs_tagver + reinterpret_cast(int_u1*, msmp));
  for (Index i = 0; i < npairs; i++) {
    serialFormatVersion(NativeMachineRep(machineRep_, tvp->tag),
			NativeMachineRep(machineRep_, tvp->version));
    ++tvp;
  }

  // Pull out the source's version string.
  char* cp = reinterpret_cast(char*, ofs_sourceVersion + reinterpret_cast(int_u1*, msmp));
  sourceVersion_ = string(cp, svl);
}



// Determine whether a message that begins with a certain byte sequence
// represents a SDE message.

bool StreamDataEncoding::isExtendedFormat (const char rep[],
					   Size replen,
					   int_u1*& pfxbuf,
					   Size& pfxlen)
{
  // m2assert(replen == SDEC_MarkerLength, "Representation marker must be " + Stringize(SDEC_MarkerLength) + " bytes long.");

  // Convert the representation to a string, and reset everything associated
  // with the encoding.
  string srep(rep, replen);
  reset();
  if (srep != string(SDEMessageMarker_, SDEC_MarkerLength)) {
    // This isn't an extended format.  Probably is from an old version of
    // M2k, or somebody using useMachineRepOnly.  Decode it as a machine
    // representation.
    machineRep_ = DecodeMachineRep(srep);
    useMachineRepOnly_ = true;
    return false;
  }

  // Is extended format.  Make a buffer big enough to hold the standard prefix,
  // copy in what we have of that prefix so far, and set up to tell the
  // caller where to store the rest of it.
  freeMessageBuffer_();
  messageLength_ = sizeof(SDEMessage_0_);
  messageBuffer_ = new int_u1[messageLength_];
  memcpy(messageBuffer_, rep, SDEC_MarkerLength);
  pfxlen = messageLength_ - SDEC_MarkerLength;
  pfxbuf = static_cast(int_u1*, messageBuffer_) + SDEC_MarkerLength;
  return true;
}



// Given the whole prefix, expand the buffer to hold the whole message, and
// tell the caller where to store the rest of it.

void StreamDataEncoding::extendMessageFromPrefix (int_u1*& mbuf, Size& mlen)
{
  SDEMessage_1_* msmp = reinterpret_cast(SDEMessage_1_*, messageBuffer_);

  // Get the length of the whole message in native byte order.
  Size wmlen = NetworkToNativeMachineRep(msmp->length_);

  // Allocate a new message, copy over the prefix, and install it in
  // place of the old buffer.
  int_u1* nmsg = new int_u1[wmlen];
  memcpy (nmsg, messageBuffer_, messageLength_);
  freeMessageBuffer_();
  messageBuffer_ = nmsg;
  msmp = reinterpret_cast(SDEMessage_1_*, messageBuffer_);

  // Get pointers to show the caller where to store the remainder of
  // the mssage (after what we already have).
  mlen = wmlen - messageLength_;
  mbuf = nmsg + messageLength_;

  // Correct the message length in anticipation of getting the rest.
  messageLength_ = wmlen;
}



// Store the data encoding information into a binary-format message to be
// transmitted to a consumer.

void StreamDataEncoding::setupMessageBuffer (int_u1*& mbuf,
						Size& mlen)
{
  if (useMachineRepOnly_) {
    // Use old-style messages, which contain only the machine representation.
    freeMessageBuffer_();
    messageLength_ = SDEC_MarkerLength;
    messageBuffer_ = new int_u1[messageLength_];
    memset(messageBuffer_, 0, messageLength_);

    MachineRep_e mrep = NativeMachineRep();
    string s = EncodeMachineRep(mrep);
    int sl = s.length();
    for (Index i = 0; i < messageLength_; i++) {
      messageBuffer_[i] = static_cast(int_u1, ((i < Index(sl)) ? s[i] : ' '));
    }
  } else {
    // Pack the message into a SDEMessage_V_-format buffer.
    packMessage_1_();
  }
  mbuf = messageBuffer_;
  mlen = messageLength_;
  return;
}



// Having read an entire SDE message, transfer its encoding information
// over into this object.

void StreamDataEncoding::setFromMessage ()
{
  reset();
  const SDEMessage_0_* msmp = reinterpret_cast(SDEMessage_0_*, messageBuffer_);
  if (1 == msmp->versionNumber_) {
    unpackMessage_1_();
    return;
  }
  //throw MachineRepException("Unrecognized format " + Stringize(msmp->versionNumber_)
  throw runtime_error("Unrecognized format " + Stringize(msmp->versionNumber_)
  
			    + " of StreamDataEncoding message");
}

PTOOLS_END_NAMESPACE 


