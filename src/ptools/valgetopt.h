#ifndef VALGETOPT_H_

#include "ocval.h"

PTOOLS_BEGIN_NAMESPACE

// Take a list of command line args and parse them into short options,
// long options and arguments.
//
// For example:
//
// usage: valgetopt_ex [-a] [-b=1] [--anon] [--bitnumber=1]
// 
// To parse this line, you pass in the command line arguments and two
// tables which describe the kind of options we want.  The first table
// lists all the short arguments and their "potentail" values (None
// indicates it is just a switch).  Note that the table keys do NOT
// have the - in it.  The second table lists the long arguments.  For
// example, in C++ code:
//
// // -a has no option, -b takes an int (i.e. -b=1 or -b 1)
// Tab short_args="{ 'a': None, 'b':1 }"; 
//
// // --anon  is just a switch with no value, --bitnumber=1 or --bitnumber 1
// Tab long_args ="{ 'anon': None, 'bitnumber': 1 }"  
// 
// // Typical usage is to ignore the program name in argv[0]
// Arr res = ValGetOpt(argc-1, argv+1, short_args, long_args);
// Tab& opts = res[0];  // The options in a table
// Arr& args = res[1];  // The arguments in order
//
// The result is a list of two elements: 
// (1) the first element is a table with all the command-line options 
//     encountered, along with their respective string values.
// (2) The second element is a list of required string
//     arguments in order.  For example, consider the invocation:

// % valgetopt -b 14 --anon arg1 argument2
//
// Arr res = ValGetOpt(argc-1, argv+1, short_args, long_args);
// cerr << res[0] << endl;  // Prints { '-b':'14', '--anon': None }
// cerr << res[1] << endl;  // Prints ['arg1', 'argument2']
// 
// /// Usage of these tables might look like:
// bool anonymous_user = false;
// int_4 bit_number = -1;
//
// It ii(opts);
// while (ii()) {
//   const Val& key   = ii.key();
//         Val& value = ii.value();
//   if      (key=="--anon" || key=="-a")       anonymous_user = true;
//   else if (key=="--bitnumber" || key=="-b")  bit_number = value;
// }
//
// If there's any problems (unexpected options, looking for value at
// the end of the line), then a ParseError is thrown (which usually
// means you should put up a usage error and immediately exit).

class ParseError : public runtime_error {
public:
  ParseError(const char* sp) : runtime_error(sp) { }
  ParseError(const string& s) : runtime_error(s.c_str()) { }
}; // ParseError

// Parse options, throw ParseError is any unexpected options 
inline Arr ValGetOpt (int argc, char**argv, 
		      const Tab& short_args, const Tab& long_args);


// Helper function: Once we've found a - or --, parse it up.  Get the option
// and it's associated values (if there are any).  We pass ii around
// because we may want to skip ahead to the next option.
inline void ValParseOption_ (int argc, char**argv, int& ii,
			     const string& full_arg,
			     const Tab& args_lookup_table, 
			     Tab& result, const string& dashes)
{
  // See if this argument is even listed in the table, but have to
  // make sure it doesn't have an = in it.  Parse it if there
  // is an equal in it.
  string::size_type where_equals = full_arg.find('=');
  string arg, value;
  if (where_equals==string::npos) { // no =, no need to break up
    arg   = full_arg;
    value = ""; 
  } else {
    arg   = full_arg.substr(0, where_equals);
    value = full_arg.substr(where_equals+1, string::npos);
  }

  // parsed arg in table?
  if (!args_lookup_table.contains(arg)) {
    throw ParseError("Option "+arg+" not listed in argument list");
  }

  // Assertion: in table, so look up its prototype value
  Val kind = args_lookup_table(arg);
  if (kind==None) { 
    // No arguments list in table, so just take 'as-is': a true switch    
    if (where_equals!=string::npos) {
      throw ParseError("= on a option "+arg+" which takes no values");
    } else {
      result[dashes+arg] = None;
    }
  } else {
    // Expect arguments ... = says right here, or next one?
    if (where_equals==string::npos) { // no =, so arg in next word
      ii++;
      if (ii>=argc) {
	throw ParseError("Expecting value for last option ("+arg+"), but not enough entries on the line");
      }
      string rest = argv[ii];
    } else {
      // the argument is just after the '=' in the string.
      // TODO: Allow "empty" value as an option I guess?
      result[dashes+arg] = value;
    }
  }
}


// implementation
inline Arr ValGetOpt (int argc, char**argv, 
		      const Tab& short_args, const Tab& long_args)
{
  Tab opts;  // Will return a Tab of options
  Arr args;  // List of args

  int ii=0;
  while (ii<argc) {
    string arg = argv[ii];
    if (arg[0]=='-') { 
      // Either short or long option
      if (arg.length()>2 && arg[1]=='-') { // Long option
	ValParseOption_(argc, argv, ii,
			arg.substr(2,string::npos), long_args, opts, "--");
      } else { // short option
	ValParseOption_(argc, argv, ii,
			arg.substr(1,string::npos), short_args, opts, "-");
      } 
    }
    
    // No -, must be a plain old arg
    else {
      args.append(arg);
    }
    
    // Go to next argument
    ii+=1;
  }

  // Build tuple of return value
  Arr ret;
  ret.append(opts);
  ret.append(args);
  return ret;
}


PTOOLS_END_NAMESPACE

#define VALGETOPT_H_
#endif // VALGETOPT_H_
