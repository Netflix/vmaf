MATLAB Compiler

1. Prerequisites for Deployment 

. Verify the MATLAB Runtime is installed and ensure you    
  have installed version 9.0 (R2015b).   

. If the MATLAB Runtime is not installed, do the following:
  (1) enter
  
      >>mcrinstaller
      
      at MATLAB prompt. The MCRINSTALLER command displays the 
      location of the MATLAB Runtime installer.

  (2) run the MATLAB Runtime installer.

Or download the Macintosh version of the MATLAB Runtime for R2015b 
from the MathWorks Web site by navigating to

   http://www.mathworks.com/products/compiler/mcr/index.html
   
   
For more information about the MATLAB Runtime and the MATLAB Runtime installer, see 
Package and Distribute in the MATLAB Compiler documentation  
in the MathWorks Documentation Center.    


NOTE: You will need administrator rights to run MCRInstaller. 


2. Files to Deploy and Package

Files to package for Standalone 
================================
-run_cvqm.sh (shell script for temporarily setting environment variables and executing 
 the application)
   -to run the shell script, type
   
       ./run_cvqm.sh <mcr_directory> <argument_list>
       
    at Linux or Mac command prompt. <mcr_directory> is the directory 
    where version 9.0 of the MATLAB Runtime is installed or the directory where 
    MATLAB is installed on the machine. <argument_list> is all the 
    arguments you want to pass to your application. For example, 

    If you have version 9.0 of the MATLAB Runtime installed in 
    /mathworks/home/application/v90, run the shell script as:
    
       ./run_cvqm.sh /mathworks/home/application/v90
       
    If you have MATLAB installed in /mathworks/devel/application/matlab, 
    run the shell script as:
    
       ./run_cvqm.sh /mathworks/devel/application/matlab
-MCRInstaller.zip 
   -if end users are unable to download the MATLAB Runtime using the above  
    link, include it when building your component by clicking 
    the "Runtime downloaded from web" link in the Deployment Tool
-The Macintosh bundle directory structure cvqm.app 
   -this can be gathered up using the zip command 
    zip -r cvqm.zip cvqm.app
    or the tar command 
    tar -cvf cvqm.tar cvqm.app
-This readme file 

3. Definitions

For information on deployment terminology, go to 
http://www.mathworks.com/help. Select MATLAB Compiler >   
Getting Started > About Application Deployment > 
Deployment Product Terms in the MathWorks Documentation 
Center.


4. Appendix 

A. Mac systems:
In the following directions, replace MCR_ROOT by the directory where the MATLAB Runtime 
   is installed on the target machine.

If the environment variable DYLD_LIBRARY_PATH is undefined, set it to the concatenation 
   of the following strings:

    MCR_ROOT/v90/runtime/maci64:
    MCR_ROOT/v90/sys/os/maci64:
    MCR_ROOT/v90/bin/maci64

If it is defined, set it to the concatenation of these strings:

    ${LD_LIBRARY_PATH}: 
    MCR_ROOT/v90/runtime/maci64:
    MCR_ROOT/v90/sys/os/maci64:
    MCR_ROOT/v90/bin/maci64

   For more detail information about setting the MATLAB Runtime paths, see Package and 
   Distribute in the MATLAB Compiler documentation in the MathWorks Documentation Center.


     
        NOTE: To make these changes persistent after logout on Linux 
              or Mac machines, modify the .cshrc file to include this  
              setenv command.
        NOTE: The environment variable syntax utilizes forward 
              slashes (/), delimited by colons (:).  
        NOTE: When deploying standalone applications, it is possible 
              to run the shell script file run_cvqm.sh 
              instead of setting environment variables. See 
              section 2 "Files to Deploy and Package".    



5. Launching of application using Macintosh finder.

If the application is purely graphical, that is, it doesn't read from standard in or 
write to standard out or standard error, it may be launched in the finder just like any 
other Macintosh application.



