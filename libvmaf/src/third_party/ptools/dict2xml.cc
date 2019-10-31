
#include "xmltools.h"

// Tool demonstrating how to convert from Python Dictionaries to XML

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
    cerr << "usage:" << argv[0] << "[[input.pythondict] output.xml]" << endl;
    cerr << "         With no options, this reads stdin and output to stdout" << endl;
    exit(1);
  }
  
  Val v;
  if (input_stdin) {
    ReadValFromStream(cin, v);
  } else {
    ReadValFromFile(argv[1], v);  // Throw exceptions for us
  }

  int xml_options = 
    XML_STRICT_HDR  | // keep XML headers
    XML_DUMP_PRETTY | // have nesting
    XML_DUMP_STRINGS_BEST_GUESS   // Guess as to whether we need quotes
    ;
  ArrayDisposition_e arr_dis = AS_NUMERIC;

  if (output_stdout) {
    WriteValToXMLStream(v, cout, "root", xml_options, arr_dis);
  } else {
    WriteValToXMLFile(v, argv[2], "root", xml_options, arr_dis);
  }

}
