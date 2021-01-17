/*
 * cpu_info.c
 *
 *  Created on: 14.02.2018
 *      Author: thomas
 */

#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#elif __FreeBSD__
#include <sys/types.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

int getNumCores() {
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#elif MACOS
    int nm[2];
    size_t len = 4;
    uint32_t count;

    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);

    if(count < 1) {
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        if(count < 1) { count = 1; }
    }
    return count;
#elif __FreeBSD__
    int ncpu;
    size_t sz = sizeof(int);
    sysctlbyname("hw.ncpu", &ncpu, &sz, 0,0);
    return(ncpu);
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}
