
#include "sharedmem.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

PTOOLS_BEGIN_NAMESPACE

// For flexibility, allow choosing between different types
// of shared memory.  Currently only POSIX works.
// #define SYSV_PIPE_SYSTEM
#define POSIX_SHM

// Implementation using Posix Shared Memory
#if defined(POSIX_SHM)


// IMPLEMENTATION NOTE:  Under Linux 8.0 and Solaris 5.7,
// ftruncate sets all new bytes to zero, so we can "key" on zeroes.
// Under DECUnix, (from the DECUnix man page)
// "If the new length is greater than the previous length, one byte of
// zero (0x00) is written at the offset of the new length.  The space
// in between the pre- vious end-of-file and the new end-of-file is
// left as a hole; that is, no blocks are allocated to the space in
// between the previous last block and the new last block."

// What this means: There is SOME BYTE that is initialized to zero
// when the memory is mapped for the first time.  To avoid race
// conditions of memory being created and attached to too quickly, we
// write the "zero_byte" to a 1 when the data structure is done being
// initialized.

static volatile char* ZeroByte (void* memory, size_t len)
{
  volatile char* mem = (volatile char*) memory;
#if   defined(OSF1_)
  mem += len-1;   // Use just after???
#elif defined(SOLARIS_)
  mem += len-1; // All bytes inited to zero, use very last legal byte
#elif defined(LINUX_)
  mem += len-1; // All bytes inited to zero, use very last legal byte
#else
# error 6667  // Look at the man page for ftruncate ... what byte is zeroed?
#endif
  return mem;
}

bool SHMInitialized (void* memory, size_t len)
{
  return (*ZeroByte(memory, len))!=0;
}

void SHMInitialize (void* memory, size_t len)
{
  *ZeroByte(memory, len) = 1;
}


void* SHMCreate (const char* pipe_path, size_t total_bytes_needed, bool debug,
		 void* forced_addr, BreakChecker external_break,
		 int_8 micro_sleep)
{
  // Set flags to open the shared memory
  int oflags = O_RDWR | O_CREAT;// always has to be r/write so can use syncs
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

  // Create the pipe.
  int fd;
  while (1) {
    fd = shm_open(pipe_path, oflags, mode);
    if (fd == -1) {
      if (errno==EINTR) continue;
      perror("shm_open");
      return 0;
    }
    else break;
  }
  if (debug) {
    cerr << ".. seemed to open up shared memory okay ..." << endl;
  }

  // Make memory right size: Only creator needs to truncate it
  // cerr << "Creating pipe with ftruncate " << endl;
  while (1) {
    if (external_break && external_break()) {
      throw runtime_error("External break received: Aborting creating "+string(pipe_path));
    }
    if (ftruncate(fd, total_bytes_needed)) {
      if (errno == EAGAIN || errno == EINTR) {
	if (debug) {
	  cerr << "EAGAIN on ftruncate" << endl;
	}
        usleep(micro_sleep);
        continue;
      } else {
        perror("ftruncate");
        return 0;
      }
    } else {
      break;
    }
  }
  if (debug) {
    cerr << " ...should have made it the right size..." << endl;
  }

  // Map it into our process
  int options = MAP_FILE|MAP_SHARED;
  if (forced_addr) options |= MAP_FIXED;
  void* ptr =  mmap(forced_addr, total_bytes_needed, PROT_READ|PROT_WRITE,
                    options, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap in SHMCreate");
    return 0;
  }
  if (debug) {
    cerr << " .. seemed to map it into memory okay ... " << endl;
  }


  /* // Lock memory ?
  while (1) {
    cerr << "Lock the memory so doesn't sync" << endl;
    int ret = mlock(ptr, total_bytes_needed);
    if (ret!=0) {
      cerr << "Hm:  Couldn't lock memory: errno = " << errno << endl;
      if (errno == EAGAIN) {
        cerr << "EAGAIN on mlock" << endl;
        usleep(100000);
        continue;
      } else {
        perror("mlock");
        cerr << "Continung after no lock ..." << endl;
        break;
        // return 0;
      }
    } else {
      cerr << "Valid locking" << endl;
      break;
    }
  }
  */

  close(fd);
  if (debug) {
    cerr << " ... seemed to close fd correctly" << endl;
  }

  return ptr;
}



void* SHMAttach (const char* pipe_path, void* forced_addr,
                 size_t& total_bytes_needed,
		 bool debug, 
		 BreakChecker external_break,
		 int_8 micro_sleep)
{
 try_again: // Race condition, may have to re-open

  // Set flags to open the shared memory
  int oflags = O_RDWR; // always has to be read/write so can use syncs
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

  // If the shared memory doesn't exist yet and we are not creating
  // it, wait for it
  int fd;
  int waittime = 0;
  while (1) {
    if (external_break && external_break()) {
      throw runtime_error("External break seen: aborting SHMAttach to:" + string(pipe_path));
    }
    fd = shm_open(pipe_path, oflags, mode);
    if (fd == -1) {
      if (errno==ENOENT || errno==EINTR) {
        usleep(micro_sleep); // 1/10 of a second or 1e5 microseconds
        if (waittime++ > 100) {
          // cerr << "...waiting for pipe " << pipe_path << " to be created ..." << endl;
          waittime=0;
        }
        continue;
      } else {
        perror("shm_open");
        return 0;
      }
    } else {
      // cerr << "Got valid file descriptor!" << endl;
      break;
    }
  }

  if (debug) {
    cerr << ".. seemed to open up shared memory okay ..." << endl;
  }

  // Make memory right size
  struct stat sta;
  if (fstat(fd, &sta)) {
    perror("fstat");
    return 0;
  }
  total_bytes_needed = sta.st_size;
  if (total_bytes_needed == 0) {
    // Can't have a pipe with zero bytes ... it was probably a race
    // condition: file not created yet, but file descriptor allocated.
    // Clean up and try again.
    close(fd);
    goto try_again;
    // return SHMAttach(pipe_path, total_bytes_needed);
  }

  // Map it into our process (possibly forcing it into an address)
  int keep_fixed = 0;
  if (forced_addr) { 
    keep_fixed=MAP_FIXED;
  }
  if (debug) {
    cerr << "BEFORE MMAP in SHMAttach: " << hex << forced_addr << endl;
  }
  void* ptr = mmap(forced_addr, total_bytes_needed, PROT_READ|PROT_WRITE,
                   MAP_FILE | MAP_SHARED | keep_fixed, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap in SHMAttach");
    return 0;
  }

  if (debug) {
    cerr << "... seemed to map it into memory okay ... " << endl;
  }

  // Can close the file descriptor once mmap is done.
  // 
  close(fd); // SOMETIMES CAN'T CLOSE????
  if (debug) {
    cerr << "... seemed to close up okay (not closing right now)" << endl;
  }

  return ptr;
}


void SHMDetach (void* memory, size_t total_bytes)
{
  int ret = munmap(memory, total_bytes);
  if (ret!=0) {
    perror("munmap");
    throw runtime_error("munmap");
  }
}

// Actually unlink the file
void SHMUnlink (const char* name, bool show_error)
{
  int ret = shm_unlink(name);
  if (ret!=0) {
    if (show_error) {
      perror("shm_unlink");
    }
    // throw runtime_error("shm_unlink");
  }
}


PTOOLS_END_NAMESPACE

#endif // POSIX_SHM


