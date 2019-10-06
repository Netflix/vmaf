#ifndef SHAREDMEMORY_H_

// Simple routines to manage shared memory.

#include "ocval.h"

PTOOLS_BEGIN_NAMESPACE

// A pointer to a function: this allows the user to pass
// in "are we aborted?" function to these routines so these
// routines that "loop and look" can use an external impetus
// to exit.
typedef bool (*BreakChecker)();

// Create a piece of shared memory with the give number of bytes.
// Returns the piece of memory (not necessarily initialized).
//
// Expert Note: as we wait for the memory to attach, we loop
// and occasionally have to back off a short while do we don't
// saturate the CPU: the micro_sleep controls how long we back off.
// In case something external wishes to stop (so we don't deadlock
// in here), the backoff loop can check an external break.
// if it's true, it throws a runtime_error exception and does not attach.
void* SHMCreate (const char* pipe_path, size_t total_bytes_needed, 
		 bool debug=false, void* forced_addr=0,
		 BreakChecker external_break=0, int_8 micro_sleep=int_u8(1e5));

// Attach to a piece of shared memory: If it's not there, block and
// wait for it to be created.  Returns the shared memory (mapped into
// the current address space), as well as the number of bytes in the
// shared memory segment.  
// 
// To make sure all mmaps WILL give the same address: If
// "force_memory_start_here" is NULL, then the memory segment is
// mapped into whatever mmap chooses: otherwise it must map to the
// given location: if it can't, a zero is returned.  The
// "force_memory_start_here" argument is almost always what is
// returned by SHMCreate (if it is not NULL)
//
// CAVEAT EMPTOR with forced_memory_starts_here being anything but NULL: 
// see comments at end of this file.
// 
// Expert Note: as we wait for the memory to attach, we loop
// and occasionally have to back off a short while do we don't
// saturate the CPU: the micro_sleep controls how long we back off.
// In case something external wishes to stop (so we don't deadlock
// in here), the backoff loop can check an external break.
// if it's true, it throws a runtime_error exception and does not attach.
void* SHMAttach (const char* pipe_path, void* forced_memory_start_here,
		 size_t& total_bytes_needed, bool debug=false,
		 BreakChecker external_break=0, int_8 micro_sleep=int_u8(1e5));

// Detach from a piece of shared memory and unmap it from our space.
void SHMDetach (void* memory, size_t len);

// Actually unlink the file: If it's not there, can shoose
// whether or not to call perror
void SHMUnlink (const char* name, bool show_error=true);


// After getting some shared memory, we need to see if it has been
// intialized correctly (SHMInitialize is called below) so we don't
// attach to it too quickly.  (This is to prevent the race condition
// of a reader connecting just after the memory has been mapped, but
// not initialized by the writer).
bool SHMInitialized (void* memory, size_t len);

// After creating then initializing the shared memory, we need to let
// readers know it's okay to final get it.
void SHMInitialize (void* memory, size_t len);


// COMMENTS: Getting "forced_memory_start_here" to work for you is a
// complex issue.  Older versions of Linux didn't have trouble with
// this, newer versions need to worry about the "RedHat Address
// Randomization" optimization.  To keep hackers from exploiting
// static addresses, some Linux kernels use address randomization, so
// data structures "randomly" go across memory (in particulcar where
// mmap returns, memory, etc).  For the shared memory feature to work
// with "forced_memory_starts_here", you have to turn the address
// randomization feature off.  There are many ways to do this, (google
// redhat address randomization), but the easiest way is to disable it
// for the current session is to "setarch i386 cmd" where cmd is the
// executable you start up.
// 
// But, they still may not be enough: mmap with MAP_FIXED may or may
// not work for you: if you are using data structures that HAVE to
// have the same addresses across processes, the best way to guarantee
// it currently is with mapping the address in the parent, then the
// child will get it.  The forced_memory_start_here is only supported
// if mmap with MAP_FIXED works well.

PTOOLS_END_NAMESPACE

#define SHAREDMEMORY_H_
#endif // SHAREDMEMORY_H_
