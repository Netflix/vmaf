
#include "xmltools.h"

// Tool demonstrating how to convert from XML to Python Dictionaries

int main (int argc, char **argv)
{
  bool input_stdin = false;
  bool output_stdout = false;
  if (argc==1) {
    // use as filter: stdin and stdout
    input_stdin = true;
    output_stdout = true;
  } else if (argc==2) {
    input_stdin = false;
    output_stdout = true;
  } else if (argc!=3) {
    cerr << "usage:" << argv[0] << "[[input.xml] output.pythondict]" << endl;
    cerr << "         With no options, this reads stdin and output to stdout" << endl;
    exit(1);
  }
  
  Val v;
  int xml_options = 
    XML_STRICT_HDR |     // make sure we have XML header
    XML_LOAD_DROP_TOP_LEVEL | // extra top-level container, nuke it
    XML_LOAD_EVAL_CONTENT     // Try to turn some strings into real values
    ;
  ArrayDisposition_e arr_dis = AS_NUMERIC;
           
  if (input_stdin) {
    ReadValFromXMLStream(cin, v, xml_options, arr_dis) ;
  } else {
    ReadValFromXMLFile(argv[1], v, xml_options, arr_dis);  // Throw exceptions for us
  }

  if (output_stdout) {
    WriteValToStream(v, cout); 
  } else {
    WriteValToFile(v, argv[2]);
  }

}
