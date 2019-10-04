
#include "xmltools.h"

// Simple tool showing how to read Vals from a Stream and convert them
// from Python Dictionaries to XML

int main (int argc, char **argv)
{
  if (argc!=2) { // read from filename provided
    cerr << argv[0] << " filename:  reads multiple Vals from a stream and converts each complete dict/list to XML" << endl;
    exit(1);
  }
  ifstream is(argv[1]);

  StreamValReader svr(is);
  while (!svr.EOFComing()) {
    Val v;
    try {
      svr.expectAnything(v);
    } catch (const exception& e) {
      cout << e.what() << endl;
      cout << " ... trying to continue reading from stream, probably won't work" << endl;
      continue;
    }

    // Assertion: Have read a Val (table) from the input file:
    // Convert to XML
    WriteValToXMLStream(v, cout, "top", 
			XML_STRICT_HDR |               // put XML hdr
			XML_DUMP_PRETTY |              // show nested struct
			XML_DUMP_STRINGS_BEST_GUESS    // no uncessary quotes
			);
  }
  
}
