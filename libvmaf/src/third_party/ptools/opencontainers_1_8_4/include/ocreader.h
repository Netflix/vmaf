#ifndef OCREADER_H_

// The StringReader and StreamReader class exist to help us parse
// Tables.  In OpenContainers parlance, these classes only exist to
// help us parse Python Dictionaries, but in the PicklingTools
// distributions, OpalTables and other kinds of ASCII Tables (and now XML) 
// need to be parsed, and they all share a similar codebase for getting
// characters.  The error handling codebase, for example.

#include "occircularbuffer.h"

OC_BEGIN_NAMESPACE

// A helper class that keeps track of where we are within the parsing:
// allows us to return better error messages from parsing the tabs.
class Context_ {
 public:
  // Create a Parsing Context and remember the last n lines for when
  // error messages happen
  Context_ (int keep_last_n_lines=5) : 
    contextLines_(keep_last_n_lines),
    data_(1024),
    lineNumber_(1),
    charNumber_(0) 
    { }

  // Add a character to the context
  void addChar (char c) 
  {
    // Notice the \n so we can notice when new lines begin
    if (c=='\n') {
      lineNumber_ += 1;
      charNumber_ =  0;
    }
    // Keep the last 1024 or so characters
    if (data_.full()) {
      data_.get();
    }
    data_.put(c);
    charNumber_++;
  }

  // Delete a character from the context
  void deleteLastChar () 
  {
    char& c = data_.drop();
    // Notice the \n so we can notice when new lines begin
    if (c=='\n') {
      lineNumber_ -= 1;
      // Find last \n ... if we can
      int index_of_last_newline = -1;
      for (int ii=0; ii<data_.length(); ii++) {
	if (data_.peek(data_.length()-ii-1)=='\n') {
	  index_of_last_newline = ii;
	  break;
	}
      }
      charNumber_ = (index_of_last_newline==-1) ? 80: index_of_last_newline;
    } else {
      charNumber_--;
    }
  }

  // Add from this buffer, the amount of data
  void addData (const char* buffer, int len)
  {
    for (int ii=0; ii<len; ii++) {
      addChar(buffer[ii]);
    }
  }

  // Generate a string which has the full context (the last n lines)
  string generateReport () 
  {
    string report;

    // Post processing: Create an array of the lines of input.  The
    // last line probably won't end with an newline because the error
    // probably happened in the middle of the line.
    Array<string> lines;
    string current_line;
    for (int ii=0; ii<data_.length(); ii++) {
      char c = data_.peek(ii);
      current_line.append(1, c);
      if (c=='\n') {
	lines.append(current_line);
	current_line = "";
      }
    }
    if (current_line != "") {
      current_line.append(1, '\n');
      lines.append(current_line);
    }

    // Only take the last few lines for error reporting
    int context_lines = (int(lines.length()) < contextLines_) ? 
                                lines.length() : contextLines_;
    if (context_lines) {
      int start_line = lines.length()-context_lines;

      report = "****Syntax Error on line:"+Stringize(lineNumber_)+" Last "+Stringize(context_lines)+ " line" + 
	(context_lines==1?"":"s") + " of input (lines "+
	Stringize(start_line+1)+"-"+Stringize(start_line+context_lines)+") "
	"shown below****\n";

      for (int ii=0; ii<context_lines; ii++) {
	report += "  " + lines[start_line+ii];
      }
      // Show, on last line, where!
      string cursor = "";
      cursor.append(charNumber_+1, '-');
      cursor += "^\n";
      report += cursor;
    }
    return report;
  }

  // Have everything do a syntax error the same way
  void syntaxError (const string& s) 
  { 
    string report = generateReport() + s;
    throw logic_error(report);
  }



 protected:

  // When holding the data, we make the "keeping context" operations
  // cheap, but when we need to actually supply the information, it's
  // a little more expensive, as we have to look for line breaks, etc.

  int contextLines_;  // how lines of context to hold (i.e., how much do we
                      // buffer)
  CircularBuffer<char> data_;  
                      // The current "context", i.e., that last few lines 
  int lineNumber_;    // Current line number we are on
  int charNumber_;   // current character within that line


}; // Context_


// Interface for all Readers parsing ASCII streams.  They all have a
// context for holding the current line numbers for error reporting
// purposes.
class ReaderA {
 public:
  // We can turn off the context if we are doing parsing where we
  // don't care about the context
  ReaderA (bool supports_context) :
    context_(supports_context ? new Context_ : 0) 
  { }

  void syntaxError (const char* s) 
  {
    // Syntax Error generates a full report, which is expensive:
    // you can turn it off if you are doing silent reads
    if (context_) {
      context_->syntaxError(string(s)); 
    } else {
      throw logic_error(s);
    }
  }

  void syntaxError (const string& s) 
  {
    // Syntax Error generates a full report, which is expensive:
    // you can turn it off if you are doing silent reads
    if (context_) {
      context_->syntaxError(s); 
    } else {
      throw logic_error(s);
    }
  }

  bool EOFComing () { return peekNWSChar_() == EOF; }
  virtual ~ReaderA () { delete context_; } // Needs to be virtual!
 public:
  Context_* context_; 

  virtual int getNWSChar_ ()  = 0;
  virtual int peekNWSChar_ () = 0;
  virtual int getChar_ ()     = 0;
  virtual int peekChar_ ()    = 0;
  virtual int consumeWS_ ()   = 0;
  virtual void pushback_ (int pushback_char) = 0;
}; // ReaderA


// A StringReader exists to read some ASCII tables from a string
class StringReader : public ReaderA {

 public:

  // Read input from array of char: may specify if we can just read
  // or f we have to make a copy
  StringReader (Array<char>& a, bool make_copy=false, 
		bool supports_context=true) :
    ReaderA(supports_context),
    length_(a.length()),
    current_(0)
  { 
    if (make_copy) {
      data_ = new char[length_];
      memcpy(data_, a.data(), length_);
      adopting_ = true;
    } else {
      data_ = a.data();
      adopting_ = false;
    }
  }

  // Read a C-style string: may have to make own copy
  StringReader (const char* s, int len=-1, bool make_copy=false,
		bool supports_context=true) :
    ReaderA(supports_context),
    length_(len==-1 ? strlen(s) : len),
    current_(0)
  {
    if (make_copy) {
      data_ = new char[length_];
      memcpy(data_, s, length_);
      adopting_ = true;
    } else {
      data_ = const_cast<char*>(s);
      adopting_ = false;
    }
  }

  // Read a string: may have to make own copy
  StringReader (const string& s, bool make_copy=false,
		bool supports_context=true) :
    ReaderA(supports_context),
    length_(s.length()),
    current_(0)
  {
    if (make_copy) {
      data_ = new char[length_];
      memcpy(data_, s.data(), length_);
      adopting_ = true;
    } else {
      data_ = const_cast<char*>(s.data());
      adopting_ = false;
    }
  }

  virtual ~StringReader ()  { if (adopting_) delete [] data_; }


 public:

  // ///// Data Members

  char* data_;      // pointer to start of data: not necessarily adopted
  int length_;      // length of full data: may just be buffer_.length() or not
  bool adopting_;   // Determines how we clean up: 

  int current_;   // Current place in the input stream, regardless of input 


  // Return the index of the next Non-White Space character.  This is
  // where comments are handled: comments are counted as white space.
  // The default implementation treats # and \n as comments
  virtual int indexOfNextNWSChar_ () 
  {
    const int len=length_;
    int cur = current_;
    if (cur==len) return cur;
    // Look for WS or comments that start with #
    bool comment_mode = false;
    for (; cur<len; cur++) {
      if (comment_mode) {
	if (data_[cur]=='\n') { comment_mode = false; }
	continue;
      } else {
	if (isspace(data_[cur])) continue;
	else if (data_[cur]=='#') { 
	  comment_mode = true;
	  continue;
	} else {
	  break;
	}
      }
    }
    return cur;
  }
  
  // Get a the next non-white character from input
  virtual int getNWSChar_ () 
  {
    int index = indexOfNextNWSChar_();

    // Save all chars read into 
    int old_current = current_;
    current_ = index;
    if (context_) 
      context_->addData(data_+old_current, current_-old_current);

    return getChar_();
  }

  // Peek at the next non-white character
  virtual int peekNWSChar_ () 
  {
    int index = indexOfNextNWSChar_();
    if (index>=length_) return EOF;
    unsigned char c = data_[index];
    return c;
  }

  // get the next character
  virtual int getChar_ () 
  { 
    const int len=length_; 
    if (current_==len) return EOF;

    unsigned char c = data_[current_++]; // avoid EOF/int-1 weirdness
    if (context_) context_->addChar(c);

    // Next char
    return c;
  }
  
  // look at the next char without getting it
  virtual int peekChar_ () 
  { 
    const int len=length_; 
    if (current_==len) return EOF;
    unsigned char c = data_[current_]; // avoid EOF/int-1 weirdness
    return c;
  }

  // Consume the next bit of whitespace
  virtual int consumeWS_ () 
  {
    int index = indexOfNextNWSChar_();

    int old_current = current_;
    current_ = index;
    if (context_) 
      context_->addData(data_+old_current, current_-old_current);

    if (index==length_) return EOF;
    unsigned char c = data_[index]; // avoid EOF/int-1 weirdness
    return c;
  }

  // The pushback allows just a little extra flexibility in parsing:
  // Note that we can only pushback chracters that were already there!
  virtual void pushback_ (int pushback_char) 
  {
     // EOF pushback
    if (pushback_char==EOF) {
      if (current_!=length_) {
	syntaxError("Internal Error: Attempt to pushback EOF when not at end");
      } else {
	return;
      }
    }
    if (current_<=0) {
      syntaxError("Internal Error: Attempted to pushback beginning of file");
    }
    // Normal char pushback
    else if (data_[--current_]!=pushback_char) {
      //cout << "** pushback_char" << pushback_char << " buffer" << buffer_[current_] << endl;
      syntaxError("Internal Error: Attempt to pushback diff char");
    }
    if (context_) context_->deleteLastChar();
    return;
  }

}; // StringReader





// A StreamReader exists to read in data from an input stream
class StreamReader : public ReaderA {
  
 public:

  // Open the given file, and attempt to read Vals out of it
  StreamReader (istream& is, bool supports_contextual_error_messages=true) :
    ReaderA(supports_contextual_error_messages),
    is_(is),
    cached_(132, true)
  { }

 public:

  istream& is_;
  CircularBuffer<int> cached_;

  // This routines buffers data up until the next Non-White space
  // character, ands returns what the next ws char is _WITHOUT GETTING
  // IT_.  It returns "peek_ahead" to indicate how many characters
  // into the stream you need to be to get it.

  // This is the default implementation that treats # and \n as comments
  virtual int peekIntoNextNWSChar_ (int& peek_ahead)
  {
    peek_ahead = 0;  // This marks how many characters into the stream
                     // we need to consume
    bool start_comment = false;
    while (1) {
      // Peek at a character either from the cache, or grab a new char
      // of the stream and cache it "peeking" at it.
      int c;
      if (peek_ahead >= cached_.length()) {
	c = is_.get();
	cached_.put(c);
      } else {
	c = cached_.peek(peek_ahead);
      }

      // Look at each character individually
      if (c==EOF) {
	// We never consume the EOF once we've seen it
	return c;
      } else if (start_comment) {
	peek_ahead++;
	start_comment = (c!='\n');
	continue;
      } else if (c=='#') { 
	peek_ahead++;
	start_comment = true;
	continue;
      } else if (isspace(c)) { // white and comments
	peek_ahead++;
	continue;
      } else {
	return (unsigned char)(c);
      }
    }
  }


  // Get the next Non White Space character
  virtual int getNWSChar_ () 
  {
    int peek_ahead;
    (void)peekIntoNextNWSChar_(peek_ahead);

    for (int ii=0; ii<peek_ahead; ii++) {
      int cc_ii = cached_.peek(ii);
      if (cc_ii != EOF) {  // Strange EOF char NOT in context!
	context_->addChar(cc_ii);
      }
    }
    cached_.consume(peek_ahead);
  
    return getChar_(); // This will handle syntax error message buffer for char
  }

  // Look at but do not consume the next NWS Char
  virtual int peekNWSChar_ () 
  {
    int peek_ahead;
    int c = peekIntoNextNWSChar_(peek_ahead);
    return c;
  }

  // get a char
  virtual int getChar_ () 
  { 
    int cc;
    if (cached_.empty()) {
      cc = is_.get();
    } else {
      cc = cached_.get();
    }
    if (cc!=EOF) {
      if (context_) context_->addChar(cc);
    }
    return cc;
  }
  
  virtual int peekChar_ () 
  { 
    if (cached_.empty()) {
      int c = is_.get();
      cached_.put(c);
    }
    return cached_.peek();
  }

  virtual int consumeWS_ () 
  {
    int peek_ahead;
    int c = peekIntoNextNWSChar_(peek_ahead);

    for (int ii=0; ii<peek_ahead; ii++) {
      int cc_ii = cached_.peek(ii);
      if (cc_ii != EOF) {  // Strange EOF char NOT in context!
	if (context_) context_->addChar(cc_ii);
      }
    }
    cached_.consume(peek_ahead);

    return c;
  }

  virtual void pushback_ (int pushback_char)
  {
    if (pushback_char != EOF && context_) context_->deleteLastChar();
    cached_.pushback(pushback_char); 
  }

}; // StreamReader



// A FILEPointerReader exists to read in data from an input FILE*
class FILEPointerReader : public ReaderA {
  
 public:

  // Open the given file, and attempt to read Vals out of it
  FILEPointerReader (FILE* fp, bool supports_contextual_error_messages=true) :
    ReaderA(supports_contextual_error_messages),
    fp_(fp),
    cached_(132, true)
  { }

 public:

  FILE* fp_;
  CircularBuffer<int> cached_;

  // This routines buffers data up until the next Non-White space
  // character, ands returns what the next ws char is _WITHOUT GETTING
  // IT_.  It returns "peek_ahead" to indicate how many characters
  // into the stream you need to be to get it.

  // This is the default implementation that treats # and \n as comments
  virtual int peekIntoNextNWSChar_ (int& peek_ahead)
  {
    peek_ahead = 0;  // This marks how many characters into the stream
                     // we need to consume
    bool start_comment = false;
    while (1) {
      // Peek at a character either from the cache, or grab a new char
      // of the stream and cache it "peeking" at it.
      int c;
      if (peek_ahead >= cached_.length()) {
	c = fgetc(fp_);
	cached_.put(c);
      } else {
	c = cached_.peek(peek_ahead);
      }

      // Look at each character individually
      if (c==EOF) {
	// We never consume the EOF once we've seen it
	return c;
      } else if (start_comment) {
	peek_ahead++;
	start_comment = (c!='\n');
	continue;

      } else if (c=='#') { 
	peek_ahead++;
	start_comment = true;
	continue;
      } else if (isspace(c)) { // white and comments
	peek_ahead++;
	continue;
      } else {
	return (unsigned char)(c);
      }
    }
  }


  // Get the next Non White Space character
  virtual int getNWSChar_ () 
  {
    int peek_ahead;
    (void)peekIntoNextNWSChar_(peek_ahead);

    for (int ii=0; ii<peek_ahead; ii++) {
      int cc_ii = cached_.peek(ii);
      if (cc_ii != EOF) {  // Strange EOF char NOT in context!
	context_->addChar(cc_ii);
      }
    }
    cached_.consume(peek_ahead);
  
    return getChar_(); // This will handle syntax error message buffer for char
  }

  // Look at but do not consume the next NWS Char
  virtual int peekNWSChar_ () 
  {
    int peek_ahead;
    int c = peekIntoNextNWSChar_(peek_ahead);
    return c;
  }

  // get a char
  virtual int getChar_ () 
  { 
    int cc;
    if (cached_.empty()) {
      cc = fgetc(fp_);
    } else {
      cc = cached_.get();
    }
    if (cc!=EOF) {
      if (context_) context_->addChar(cc);
    }
    return cc;
  }
  
  virtual int peekChar_ () 
  { 
    if (cached_.empty()) {
      int c = fgetc(fp_);
      cached_.put(c);
    }
    return cached_.peek();
  }

  virtual int consumeWS_ () 
  {
    int peek_ahead;
    int c = peekIntoNextNWSChar_(peek_ahead);

    for (int ii=0; ii<peek_ahead; ii++) {
      int cc_ii = cached_.peek(ii);
      if (cc_ii != EOF) {  // Strange EOF char NOT in context!
	if (context_) context_->addChar(cc_ii);
      }
    }
    cached_.consume(peek_ahead);

    return c;
  }

  virtual void pushback_ (int pushback_char)
  {
    if (pushback_char != EOF && context_) context_->deleteLastChar();
    cached_.pushback(pushback_char); 
  }

}; // FILEPointerReader




OC_END_NAMESPACE


#define OCREADER_H_
#endif // OCREADER_H_
