
#include "ocport.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


#include "ocsplit.h"

int main ()
{
  cout << "Whitespace delimits" << endl;
  const char* in[] = { "", "a", " a", "a  ", "  a ", 
		       " ab", " a b", 
		       "abde   dff   ", 
		       "  abde  \t dff\t\t ",
		       "a b c  s d   asd f asd f asd f sd    \n",
		       "a b c  s d   asd f asd f asd f sd\n",
		       0 };
  for (const char** input = &in[0]; *input!=0; input++) {
    cout << "Split('" << *input << "')" << endl;
    cout << Split(*input) << endl;
  }

  cout << "User delimits" << endl;
  const char* in1[] = { "", "a", " a", "a  ", "  a ", 
		       " ab", " a b", 
		       "abde   dff   ", 
		       "  abde  \t dff\t\t ",
		       "a b c  s d   asd f asd f asd f sd    \n",
		       "a b c  s d   asd f asd f asd f sd\n",
		       0 };
  for (const char** input = &in1[0]; *input!=0; input++) {
    cout << "Split('" << *input << "', \" \")" << endl;
    cout << Split(*input, " ") << endl;
  }

  cout << "User delimits with a" << endl;
  const char* in2[] = { "", "a", " a", "a  ", "  a ", 
		       " ab", " a b", 
		       "abde   dff   ", 
		       "  abde  \t dff\t\t ",
		       "a b c  s d   asd f asd f asd f sd    \n",
		       "a b c  s d   asd f asd f asd f sd\n",
		       0 };
  for (const char** input = &in2[0]; *input!=0; input++) {
    cout << "Split('" << *input << "', \"a\")" << endl;
    cout << Split(*input, "a") << endl;
  }

  cout << "User delimits with abbbcc" << endl;
  const char* in3[] = { "", "a", " a", "a  ", "  a ", 
		       " ab", " a b", 
		       "abde   dff   ", 
		       "  abde  \t dff\t\t ",
		       "a b c  s d   asd f asd f asd f sd    \n",
		       "a b c  s d   asd f asd f asd f sd\n",
			"123abbbcc456",
			"123abbbbcc456",
			"123abbbccc456",
			"123aabbbccc456",
			"123aabbbcc",
			"123aabbbc",
		       0 };
  for (const char** input = &in3[0]; *input!=0; input++) {
    cout << "Split('" << *input << "', \"abbbcc\")" << endl;
    cout << Split(*input, "abbbcc") << endl;
  }

  
  cout << "Splitting at end" << endl;
  string homer="60";
  Array<string> re1 = Split(homer, ";");
  cout << re1 << endl;
  Array<string> re2 = Split(homer, ";", 1);
  cout << re2 << endl;
  Array<string> re3 = Split(homer, ";", 2);
  cout << re3 << endl;
  Array<string> re4 = Split(homer, ";", -1);
  cout << re4 << endl;
 

  string protocol, host, path, port;
  ParseURL("http://www.yahoo.com/file/path",
	   protocol, host, path, port);
  cout << "protocol='" << protocol << "'" << endl;
  cout << "host='" << host << "'" << endl;  
  cout << "path='" << path << "'" << endl;
  cout << "port='" << port << "'" << endl;

  ParseURL("http://www.yahoo.com/fred.html:80",
	   protocol, host, path, port);
  cout << "protocol='" << protocol << "'" << endl;
  cout << "host='" << host << "'" << endl;  
  cout << "path='" << path << "'" << endl;
  cout << "port='" << port << "'" << endl;

  try {
    ParseURL("http://www.yahoo.com:80",
	     protocol, host, path, port);
  } catch (exception& re) {
    cerr << "ERROR: Malformed URL" << re.what() << endl;
  }
  cout << "protocol='" << protocol << "'" << endl;
  cout << "host='" << host << "'" << endl;  
  cout << "path='" << path << "'" << endl;
  cout << "port='" << port << "'" << endl;

  try {
    ParseURL("http://www.yahoo.com:/path/to/fred.html:80",
	     protocol, host, path, port);
  } catch (exception& re) {
    cerr << "ERROR: Malformed URL" << re.what() << endl;
  }
  cout << "protocol='" << protocol << "'" << endl;
  cout << "host='" << host << "'" << endl;  
  cout << "path='" << path << "'" << endl;
  cout << "port='" << port << "'" << endl;


  string hh = "abcdef";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = " abcdef";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = " \tabcdef";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = " \t\n  abcdef";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "abcdef ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "abcdef   ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "      abcdef   ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "      \r  \t  \n  abc   de f  \n  ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "   ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "  ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = " ";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "a";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "aa";
  cout << "Strip: hh='" << hh << "'    stripped:'" << Strip(hh) << "'" << endl;

  hh = "aaabbccaa";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;

  hh = "aaa";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;

  hh = "bb";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;

  hh = "b";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;


  hh = "bc";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;


  hh = "cb";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;


  hh = "c";
  cout << "Strip ab: hh='" << hh << "'    stripped:'" << Strip(hh, "ab") << "'" << endl;

}

