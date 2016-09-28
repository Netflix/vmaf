#ifndef M2STREAMDATAENC_H_

//
// ///// Authors: Peter Bigot
//
// A new class has been added to describe the representation used for data that
// has been binary serialized for storage or transport.  In the past, only the
// machine representation has been saved, recording byte order and other
// primitive type encodings.  Over time, M2k has changed the serialization used
// for higher-level data types such as EventDdata; without additional
// information it would be impossible to determine which serialization
// mechanism was used by the remote end.
// 
// The process used attempts to be backwards compatible with previous
// mechanisms as much as possible.  In the past, a four-character sequence
// representing the machineRep was sent across sockets to tell the remote end
// how data will be provided.  The new method sends a different four-character
// sequence "M2BD" (Midas 2k Binary Data), which does not represent any
// standard machine representation, to indicate that the extended format is
// being used.  Versions of M2k prior to this putback will, when receiving a
// header in the new format, fail to recognize the encoding and stop
// importing.  This is considered to be preferable to the alternative of
// silently misinterpreting data because its serialization has been changed.
// In the future, it may be possible to have M2k send old-style encodings.
// 
// The extended format contains the machine representation and a set of
// tag-and-version pairs which indicate which serialization is being used for a
// particular data representation.  It is recorded within the IMemStream binary
// stream extractor class.  For this scheme to work between versions, it is
// important that the tag values never change.  For this reason, the
// StreamDataEncoding class contains within it a list of all types that are
// currently serialized.  New tags can be added to the list without affecting
// the ability of older versions to read the tags that they understand.  If it
// does become necessary to change the tag list, the format change will have to
// be reflected in the encoding version for the StreamDataEncoding class
// itself.
// 
// Serialization format versions are recorded as monotonically increasing
// integer static constants in the class that is being serialized.  The
// method that does the deserialization asks the imemstream which provides the
// data for the serialization version used for the data type that it is
// deserializing.  If the version is later (greater) than the latest one
// recognized by the extractor method, an exception is thrown.  Otherwise, the
// extractor pulls out data using the desired serialization format.
// 
// For performance reasons, only those classes which have changed serialization
// in the past, or can reasonably be expected to change it in the future, have
// had support added to them for checking the serialization version.  In other
// classes, it is assumed that "original" serialization is being used.  This
// presents no problem until another serialization is implemented; when it
// does, then without at least a basic check older versions of M2k will not be
// able to detect that they have been sent data that they do not know how to
// decode.


#include "ocarray.h"
#include "m2convertrep.h"

PTOOLS_BEGIN_NAMESPACE


class StreamDataEncoding {
  public:

    // Enumerations for all types that have binary serializations.  Format
    // of the enumeration tag is "SFI_" <ClassName> "_" ("ie"|"bs"), where
    // "ie" indicates the format used in MemStream inserters and
    // extractors, and "bs" indicates the format used in
    // binarySerialize/binaryDeserialize methods.

    // To add a new serialization format:
    // + Append the tag SFI_ClassName_fmt to the enumeration below at the
    //   end (just before SFI_UserDefined)
    // + Declare static const int_u4 ClassName:SFV_Original = 1
    // + Declare static const int_u4 ClassName:SFV_Desc = .. for each
    //   non-original serialization version.  Document the M2k version
    //   in which the version was introduced.
    // + Declare static const int_u4 ClassName::SFV_Latest = ClassName::SFV_Desc
    //   for the most recent serialization version
    // + In component/m2opalvalue.cc:setStreamDataEncodingVersions add:
    //   sde.serialFormatVersion(StreamDataEncoding::SFI_ClassName_fmt, ClassName::SFV_Latest);
    // + In the extractor method, add:
    //   int_u4 sfv = in.serialFormatVersion(StreamDataEncoding::SFI_ClassName_fmt);
    //   if (StreamDataEncoding::SFV_Latest == sfv) {
    //     sfv = ClassName::SFV_Latest;
    //   } else if (StreamDataEncoding::SFV_Original == sfv) {
    //     // Alternatively assign a non-original format if that was more common
    //     sfv = ClassName::SFV_Original;
    //   }
    //   if (sfv > M2Time::SFV_Latest) {
    //     throw StreamDataEncodingUnrecEx("M2Time", sfv, M2Time::SFV_Latest);
    //   }
    // + Augment the extractor method to recognize multiple formats.

    // ///// Type Definitions

    enum SerialFormatIdentifier_e {
      SFI_Unknown,
      SFI_bool_ie,
      SFI_string_ie,
      SFI_M2Time_ie,
      SFI_Vector_ie,
      SFI_TimeStamp_ie,
      SFI_IndexedTimeStamp_ie,
      SFI_TimePacket_ie,
      SFI_Strings_ie,
      SFI_OpalValue_ie,
      SFI_OpalValueT_bs,
      SFI_OpalTable_ie,
      SFI_OpalTableProxy_bs,
      SFI_OpalHeader_bs,
      SFI_OpalTable_bs,
      SFI_Number_ie,
      SFI_MultiVector_ie,
      SFI_Fun_ie,
      SFI_EventData_ie,
      SFI_OpalLink_bs,

      SFI_UserDefined = 0x1000
    };


    // ///// Data Members

    // Generic versions that represent the oldest and newest formats for
    // binary serialization of a particular type.
    static const int_u4 SFV_Original;
    static const int_u4 SFV_Latest;

    
    // ///// Methods

    StreamDataEncoding ();
    ~StreamDataEncoding ();


    // Set and get the machine representation of the encoded data.

    MachineRep_e machineRep () const { return machineRep_; }
    void machineRep (MachineRep_e rep) { machineRep_ = rep; }


    // Encode the SDE parameters into a binary format to be transferred or
    // stored.

    void setupMessageBuffer (int_u1*& mbuf, Size& mlen);


    // Determine whether an incoming encoding message represents an SDE.
    // Returns "true" iff it does, and if so, returns in mbuf/mlen where
    // the remainder of the message prefix should be stored.

    bool isExtendedFormat (const char rep[], Size replen, int_u1*& mbuf, Size& mlen);


    // Extract data from the message prefix, and determine where the rest
    // of the message should be stored.

    void extendMessageFromPrefix (int_u1*& mbuf, Size& mlen);


    // Extract the encoding specification from the full message.
    
    void setFromMessage ();


    // What version should be returned when we don't have any information
    // about the encoding?  Must be either SFV_Original or SFV_Latest.

    int_u4 defaultSerialFormatVersion () const { return defaultSFV_; }
    void defaultSerialFormatLatest (bool ul)
    {
      defaultSFV_ = ul ? SFV_Latest : SFV_Original;
    }


    // Set the serialization version for tag sfi to be sfv.

    void serialFormatVersion (int_u4 sfi, int_u4 sfv)
    {
      if (sfi < sfValidVersion_.length()) {
	sfValidVersion_[sfi] = true;
	sfVersion_[sfi] = sfv;
      } else {
	for (Index i = 0; i < sfExtendedTag_.length(); i++) {
	  if (sfExtendedTag_[i] == sfi) {
	    sfExtendedVersion_[i] = sfv;
	    return;
	  }
	}
	sfExtendedTag_.append(sfi);
	sfExtendedVersion_.append(sfv);
      }
      return;
    }
    
      
    // Extract the serialization version for the given tag, or return the
    // default if we have no information about serialization for that tag.
    
    int_u4 serialFormatVersion (int_u4 sfi) const
    {
      if (sfi < sfValidVersion_.length()) {
	return sfValidVersion_[sfi] ? sfVersion_[sfi] : defaultSFV_;
      }
      for (Index i = 0; i < sfExtendedTag_.length(); i++) {
	if (sfExtendedTag_[i] == sfi) {
	  return sfExtendedVersion_[i];
	}
      }
      return defaultSFV_;
    }


    // Clear all encoding information (representation, tags, etc.)

    void reset ();


    // Determine whether this encoding specification depends solely
    // on the machine representation (backwards compatibility).
    
    void useMachineRepOnly (bool mro) { useMachineRepOnly_ = mro; }
    bool useMachineRepOnly () const { return useMachineRepOnly_; }


  private:
    
    // ///// Data members

    // If true, outgoing messages consist only of the machine rep, and
    // this class's settings were obtained from only a machine rep.

    bool useMachineRepOnly_;


    // Which version do we return when lacking information?

    int_u4 defaultSFV_;


    // Version number used in a message

    int_u1 versionNumber_;

    // sfValidVersion_ is an array of flags indicating whether we have a
    // valid version number for a low-valued tag; if so version is stored
    // in sfVersion_.  Arrays are sized in constructor, must be the same
    // size, and are always full.

    Array<bool> sfValidVersion_;
    Array<int_u4> sfVersion_;


    // For larger-valued tags, the following two arrays are maintained in
    // tandem, containing the tags and their corresponding serialization
    // version.

    Array<int_u4> sfExtendedTag_;
    Array<int_u4> sfExtendedVersion_;


    // The version of M2k from which a message was received.

    string sourceVersion_;


    // The machine representation of the underlying memory.

    MachineRep_e machineRep_;


    // The buffer containing the serialized version of this class.

    int_u1* messageBuffer_;
    Size messageLength_;


    // ///// Methods


    // Block attempts to copy or assign StreamDataEncoding objects: there
    // is a dynamically allocated message buffer in them which would be
    // lost if copying were allowed.

    StreamDataEncoding (const StreamDataEncoding& rhs);
    StreamDataEncoding& operator= (const StreamDataEncoding& rhs);


    // Free the message buffer, if there is one.

    void freeMessageBuffer_ ()
    {
      if (0 != messageBuffer_) {
	delete[] messageBuffer_;
      }
      messageBuffer_ = 0;
    }


    // Methods that build and read each version of the serialization
    // message formats.

    void packMessage_1_ ();
    void unpackMessage_1_ ();

};

PTOOLS_END_NAMESPACE 

#define M2STREAMDATAENC_H_
#endif // M2STREAMDATAENC_H_
