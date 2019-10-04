#ifndef OCEXCEPTION_H_

// ///////////////////////////////////////////// Includes

OC_BEGIN_NAMESPACE

// Mimic some of the STL exceptions
class exception {
  public:
    exception () { }
    virtual ~exception () { }
    virtual const char* what () const { return "generic exception"; }   
}; // exception

class logic_error : public exception {
  public:
    logic_error (const string& what_arg) :
      str_(what_arg)
    { }
    virtual ~logic_error () { }
    virtual const char* what () const { return str_.c_str(); }
  protected:
    string str_;
}; // logic_error

class length_error : public logic_error {
  public:
    length_error (const string& what_arg) :
      logic_error(what_arg) { }
}; // logic_error

class out_of_range : public logic_error {
  public:
    out_of_range (const string& what_arg) :
      logic_error(what_arg) { }
}; // out_of_range

class runtime_error : public exception {
  public:
    runtime_error (const string& what_arg) :
      str_(what_arg)
    { }
    virtual ~runtime_error () { }
    virtual const char* what () const { return str_.c_str(); }
  protected:
    string str_;
}; // runtime_error

inline void OCThrowRangeEx (const char* routine, unsigned int i,unsigned high)
{
  //string err_msg = string(routine)+" trying to access index:"+
  //OCIntToString(i)+" for a string of length:"+OCIntToString(high);
  //throw out_of_range(err_msg);
  throw out_of_range(routine);
}

OC_END_NAMESPACE

#define OCEXCEPTION_H_
#endif // OCEXCEPTION_H_
