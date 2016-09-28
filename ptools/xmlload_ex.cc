
#include "xmltools.h"

// Example showing how to read from an XML file and convert to VAL

int main (int argc, char **argv)
{
  if (argc!=2) { // read from filename provided
    cerr << argv[0] << " filename:  reads XML from a stream and converts to dicts" << endl;
    exit(1);
  }
  ifstream is(argv[1]);
  Val v;
  ReadValFromXMLStream(is, v, 
		 XML_STRICT_HDR |     // want strict XML hdr
	         XML_LOAD_DROP_TOP_LEVEL | // XML _needs_ top-level, vals dont
		 XML_LOAD_EVAL_CONTENT
		);   // want real values, not strings
  v.prettyPrint(cout);

  return 0;
}
