cmake_minimum_required(VERSION 3.5)
project(vmaf VERSION 1.3.15 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
find_package(Threads)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

macro(target_test_compiler_flags TARGET C_OR_CXX)
    if(NOT (${C_OR_CXX} STREQUAL "C" OR ${C_OR_CXX} STREQUAL "CXX" OR ${C_OR_CXX} STREQUAL "CCXX"))
        message(AUTHOR_WARNING "In ${CMAKE_CURRENT_LIST_FILE} target_test_compiler_flags\n"
            "${C_OR_CXX} does not match C or CXX\n"
            "Defaulting to C")
    endif()

    foreach(flag ${ARGN})
        string(REGEX REPLACE "[^A-Za-z0-9]" "_" flag_var "${flag}")
        if(${C_OR_CXX} MATCHES "CXX")
            set(test_cxx_flag "CXX_FLAG${flag_var}")
            check_cxx_compiler_flag(${flag} "${test_cxx_flag}")
            if(${test_cxx_flag})
                target_compile_options(${TARGET} PUBLIC ${flag})
            endif()
        endif()
        if(${C_OR_CXX} STREQUAL "C" OR ${C_OR_CXX} STREQUAL "CCXX" OR NOT ${C_OR_CXX} MATCHES "CXX")
            set(test_c_flag "C_FLAG${flag_var}")
            check_c_compiler_flag(${flag} "${test_c_flag}")
            if(${test_c_flag})
                target_compile_options(${TARGET} PUBLIC ${flag})
            endif()
        endif()
    endforeach()
endmacro()

find_library(RT_LIB rt)
find_library(M_LIB m)

# Ptools Section

add_library(PTOOLS_COM_OBJS OBJECT
    src/ptools/m2pythontools.cc
    src/ptools/valpython.cc
    src/ptools/midassocket
    src/ptools/valprotocol2.cc
    src/ptools/m2ser.cc
    src/ptools/m2streamdataenc.cc
    src/ptools/m2convertrep.cc
    src/ptools/timeconv.cc)

target_include_directories(PTOOLS_COM_OBJS PUBLIC src/ptools/opencontainers_1_8_4/include)
target_compile_definitions(PTOOLS_COM_OBJS PUBLIC -DOC_NEW_STYLE_INCLUDES -D_REENTRANT)
target_test_compiler_flags(PTOOLS_COM_OBJS CXX -Wall -Wextra -Wno-deprecated -fpic -O -fno-strict-aliasing)

add_library(ptools $<TARGET_OBJECTS:PTOOLS_COM_OBJS>)
target_link_libraries(ptools Threads::Threads)

# End Of Ptools Section
# libsvm Section

add_library(svm src/libsvm/svm.cpp)
target_test_compiler_flags(svm CXX -w -Wconversion -fPIC)
if(MSVC)
    target_test_compiler_flags(svm CXX -nologo -EHsc)
    target_compile_definitions(svm PUBLIC -D_CRT_SECURE_NO_DEPRECATE)
    target_sources(svm src/libsvm/svm.def)
    set_property(TARGET svm OUTPUT_NAME libsvm)
    add_executable(svm-toy src/libsvm/svm-toy/windows/svm-toy.cpp)
    target_link_libraries(svt-toy svm user32 gdi32 comdlg32)
endif()

set_target_properties(svm PROPERTIES SOVERSION 2)

add_executable(svm-train src/libsvm/svm-train.c)
target_link_libraries(svm-train svm)
add_executable(svm-predict src/libsvm/svm-predict.c)
target_link_libraries(svm-predict svm)
add_executable(svm-scale src/libsvm/svm-scale.c)
target_link_libraries(svm-scale svm)

if(UNIX AND NOT APPLE AND M_LIB)
    target_link_libraries(svm-train m)
    target_link_libraries(svm-predict m)
endif()

# End of libsvm Section
# libvmaf Section

add_compile_options(-fPIC -pedantic)

set_source_files_properties(src/libvmaf/src/common/convolution_avx.c PROPERTIES COMPILE_FLAGS -mavx)

add_library(vmaf
    $<TARGET_OBJECTS:PTOOLS_COM_OBJS>
    src/libvmaf/src/adm.c
    src/libvmaf/src/adm_tools.c
    src/libvmaf/src/all.c
    src/libvmaf/src/ansnr.c
    src/libvmaf/src/ansnr_tools.c
    src/libvmaf/src/common/alignment.c
    src/libvmaf/src/common/alloc.c
    src/libvmaf/src/common/blur_array.c
    src/libvmaf/src/common/convolution_avx.c
    src/libvmaf/src/common/convolution.c
    src/libvmaf/src/common/cpu.c
    src/libvmaf/src/common/file_io.c
    src/libvmaf/src/common/frame.c
    src/libvmaf/src/cpu_info.c
    src/libvmaf/src/iqa/convolve.c
    src/libvmaf/src/iqa/decimate.c
    src/libvmaf/src/iqa/math_utils.c
    src/libvmaf/src/iqa/ssim_tools.c
    src/libvmaf/src/moment.c
    src/libvmaf/src/motion.c
    src/libvmaf/src/ms_ssim.c
    src/libvmaf/src/psnr.c
    src/libvmaf/src/psnr_tools.c
    src/libvmaf/src/ssim.c
    src/libvmaf/src/svm.cpp
    src/libvmaf/src/vif.c
    src/libvmaf/src/vif_tools.c
    src/libvmaf/src/combo.c
    src/libvmaf/src/vmaf.cpp
    src/libvmaf/src/darray.c
    src/libvmaf/src/libvmaf.cpp
    src/libvmaf/src/pugixml/pugixml.cpp)

target_include_directories(vmaf PUBLIC src/libvmaf/src src/libvmaf/src/common src/ptools src/ptools/opencontainers_1_8_4/include)
target_compile_definitions(vmaf PUBLIC -DOC_NEW_STYLE_INCLUDES -D_REENTRANT)
set_target_properties(vmaf PROPERTIES PUBLIC_HEADER src/libvmaf/src/libvmaf.h)

if(UNIX AND NOT APPLE AND M_LIB)
    target_link_libraries(vmaf m)
endif()

set(PREFIX "${CMAKE_INSTALL_PREFIX}")
configure_file(src/libvmaf/libvmaf.pc.in libvmaf.pc @ONLY)

add_executable(vmafossexec src/libvmaf/src/main.cpp)
target_link_libraries(vmafossexec Threads::Threads vmaf)

add_executable(vmaf_exec src/libvmaf/src/vmaf_main.c)
target_link_libraries(vmaf_exec Threads::Threads vmaf)
set_target_properties(vmaf_exec PROPERTIES OUTPUT_NAME vmaf)

add_executable(psnr src/libvmaf/src/psnr_main.c)
target_link_libraries(psnr vmaf)

add_executable(moment src/libvmaf/src/moment_main.c)
target_link_libraries(moment vmaf)

add_executable(ssim src/libvmaf/src/ssim_main.c)
target_link_libraries(ssim vmaf)

add_executable(ms_ssim src/libvmaf/src/ms_ssim_main.c)
target_link_libraries(ms_ssim vmaf)

install(TARGETS vmaf
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libvmaf.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(DIRECTORY model DESTINATION ${CMAKE_INSTALL_DATADIR})

# Although it's possible to add an "uninstall" target by reading the install_manifest.txt file that gets generated after `make install`
# It's generally not recommended by cmake.
#uninstall:
#    rm -f $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIBVMAF)
#    rm -f $(DESTDIR)$(INSTALL_PREFIX)/include/libvmaf.h
#    rm -fr $(DESTDIR)$(INSTALL_PREFIX)/share/model
#    rm -f $(DESTDIR)$(INSTALL_PREFIX)/lib/pkgconfig/libvmaf.pc

# End of libvmaf Section
