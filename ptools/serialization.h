#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

PTOOLS_BEGIN_NAMESPACE

// Different types of serialization: Notice that this is reasonably
// backwards compatible with previous releases, and 0 and 1 still
// correspond to "Pickling Protocol 0" and "No serialization".  Now,
// the value 2 becomes "Pickling Protocol 2".
enum Serialization_e { 
  SERIALIZE_SEND_STRINGS_AS_IS_WITHOUT_SERIALIZATION = 1, // No serialization at all: Dumps as a strings, reads back as a string
  SERIALIZE_PYTHON_PICKLING_PROTOCOL_0 = 0,   // Older, slower, more compatible
  SERIALIZE_PYTHON_PICKLING_PROTOCOL_2 = 2,   // Newer, faster serialization
  SERIALIZE_PYTHON_PICKLING_PROTOCOL_2_AS_PYTHON_2_2_WOULD = -2,
  SERIALIZE_MIDAS2K_BINARY = 4,
  SERIALIZE_OPENCONTAINERS = 5,
  SERIALIZE_PYTHONTEXT = 6,   // alias
  SERIALIZE_PYTHONPRETTY = 7, // ... alias to indicate printing Python dicts
  SERIALIZE_OPALPRETTY = 8,   // ... print an OpalTable pretty
  SERIALIZE_OPALTEXT = 9,     // ... print as Opal WITHOUT pretty indent

  SERIALIZE_TEXT = 6,         // Will stringize on DUMP, Eval on LOAD
  SERIALIZE_PRETTY = 7,       // Will prettyPrint on DUMP, Eval on LOAD

  SERIALIZE_P2_OLDIMPL = -222, // Older implementations of loader
  SERIALIZE_P0_OLDIMPL = -223, // Older implementation of loader

  // Aliases
  SERIALIZE_NONE   = SERIALIZE_SEND_STRINGS_AS_IS_WITHOUT_SERIALIZATION,
  SERIALIZE_P0     = SERIALIZE_PYTHON_PICKLING_PROTOCOL_0, 
  SERIALIZE_P2     = SERIALIZE_PYTHON_PICKLING_PROTOCOL_2, 
  SERIALIZE_M2K    = SERIALIZE_MIDAS2K_BINARY,
  SERIALIZE_OC     = SERIALIZE_OPENCONTAINERS, 

  // Older versions of Python 2.2.x specificially don't "quite" work with
  // serialization protocol 2: they do certain things wrong.  Before we
  // send messages to servers and clients, we need to tell them we are
  // using an older Python that does things slightly differently.
  SERIALIZE_P2_OLD = SERIALIZE_PYTHON_PICKLING_PROTOCOL_2_AS_PYTHON_2_2_WOULD
};

PTOOLS_END_NAMESPACE

#endif // SERIALIZATION_H_
