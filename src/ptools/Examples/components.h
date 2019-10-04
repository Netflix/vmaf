
#include "ocval.h"
#include "octhread.h"
#include "occq.h"
#include <iostream>
using namespace std;

// The basics of a component based, threaded framework for processing
// data


class Component {
public:
  // All components have a dictionary of attributes
  Val attrs;

  // Component Constructor
  Component (const string& name) :
    attrs(Locked(new Tab("{'status':'stopped'}"))),
    name_(name)
  { }

  // Destructor
  virtual ~Component () { }

  // Start the component running.  Returns immediately if already
  // running
  virtual void start ()
  {
    if (get("status")=="started")
      return;

    thread_ = new OCThread(name_, false); 
    thread_->start(&mainloop_, this);
    set("status", "started");
  }
  
  // Tell the component to stop, but don't WAIT for it
  virtual void stop ()
  {
    TransactionLock tl(attrs);
    string status = attrs("status");
    if (status=="stopping") return;
    if (status=="stopped") return;
    if (status=="started") {
      attrs["status"] = "stopping";
    }
    return;
  }
  
  // Blocking wait, wait for the component to stop.
  // Immediately returns if the component is already stopped
  virtual void wait ()
  {
    if (get("status")=="stopped") 
      return;

    // Does the join, which waits for it.
    delete thread_;  
    thread_ = 0;
    set("status", "stopped");
  }
  
  virtual void connect (CQ* in, CQ* out) 
  { input_ = in; output_ = out; }

  // Set an attribute
  void set (const string& attr_name, const Val& value)
  {
    TransactionLock tl(attrs);
    attrs[attr_name] = value;
  }
   
  // Get the value of an attribute:  If it's not in the table,
  // return the default
  Val get (const string& attr_name, const Val& defalt=None)
  {
    TransactionLock tl(attrs);
    Tab& t = attrs;
    if (t.contains(attr_name))
      return t(attr_name);
    else 
      return defalt;
  }
  
  
protected:
  OCThread* thread_;  // adopted thread
  
  static void* mainloop_ (void* data) 
  {
    Component* c = (Component*)data;
    c->loop_();
  }

  // The main loop
  virtual void loop_ ()
  {
    startup_();
    while (!isDone_()) {
      pre_();
      work_();
      post_();
    }
    rundown_();
  }

  virtual void startup_ () 
  { 
    //cerr << "Starting main loop of " << name_ << endl;
    iterations_ = 0;
    TransactionLock tl(attrs);
    Tab& t = attrs;
    if (t.contains("iter")) 
      final_ = t("iter");
    else
      final_ = 1000; // some default
  }
  virtual void pre_ ()  { }
  virtual void work_ () { }
  virtual void post_ ()  
  { 
    iterations_++;
  }
  virtual void rundown_ () { }
  virtual bool isDone_ ()
  {
    string status = get("status");
    if (status=="stopped" || status=="stopping") 
      return true;
    if (iterations_==final_)
      return true;
    else 
      return false;
  }

  // Input and output
  CQ* input_;
  CQ* output_;

  // Number of iterations to run
  int_8 iterations_; // current
  int_8 final_;      // last

  // The name
  string name_;

}; // Component


class Generator : public Component {

public:
  
  Generator (const string& name) :
    Component(name)
  { }

  virtual bool generate (Val& v) = 0;

protected:
  virtual void work_ () 
  {
    Val packet;
    if (generate(packet)) 
      output_->enq(packet);
  }

}; // Generator


class Transformer : public Component {
public:
  
  Transformer (const string& name) : 
    Component(name) 
  { }

  virtual bool transform (const Val& in, Val& out) = 0;

protected:
  virtual void work_ ()
  {
    Val in = input_->deq();
    Val out;
    if (transform(in, out))
      output_->enq(out);
  }
  
}; // Transformer


class Analyzer : public Component {
public:
  Analyzer (const string& name) :
    Component(name)
  { }
  
  virtual void analyze (const Val& in) = 0;

protected:
  virtual void work_ ()
  {
    Val in = input_->deq();
    analyze(in);
  }
  
}; // Analyzer

class Constant : public Generator {
public:
  Constant (const string& name) : Generator(name), alreadyReported_(false) { }

  virtual bool generate (Val& packet)
  {
    // Get attrs
    int length = get("DataLength", 1024);

    if (!alreadyReported_) {
      //cerr << "Generating packets with DataLength (Samples per packet):" << length << endl;
      alreadyReported_ = true;
    }

    // Create packet
    packet = Locked(new Tab("{'HEADER':{}, 'DATA':None }"));
    Array<complex_8>& a = packet["DATA"] = new Array<complex_8>(length);
    a.expandTo(length);
    complex_8* output = a.data();
    
    // Compute
    for (int jj=0; jj<2; jj++) {
      for (int ii=0; ii<length; ii++) {
	output[ii] = length*jj;
      }
    }
    return true;
  }

protected:
  bool alreadyReported_;
  
}; // Constant


class Mult : public Transformer {
public:
  Mult (const string& name) : Transformer(name) { }

  virtual bool transform (const Val& inp, Val& outp)
  {
    // Get attrs
    real_8 multiplier = get("Mult", 10);

    // Unravel input packet
    Tab& packet = inp;
    complex_8* in = 0;
    int length = 0;
    {
      TransactionLock tl(inp);
      Array<complex_8>& a = packet("DATA");
      in = a.data();
      length = a.length();
    }

    // Create Output packet
    outp = Locked(new Tab("{ 'HEADER': {}, 'DATA': None }"));
    Array<complex_8>& a = outp["DATA"] = new Array<complex_8>(length);
    a.expandTo(length);

    // Compute
    complex_8* out = a.data();
    for (int ii=0; ii<length; ii++) {
      out[ii] = in[ii] * multiplier;
    }
    
    return true;
  }
}; // Mult

class MaxMin : public Analyzer {
public:
  MaxMin (const string& name) : Analyzer(name) { }

  virtual void analyze (const Val& in)
  {
    // Unravel input packet
    Tab& packet = in;
    complex_8* data = 0;
    int length = 0;
    {
      TransactionLock tl(in);
      Array<complex_8>& a = packet("DATA");
      data = a.data();
      length = a.length();
    }

    // Compute
    real_4 maxi = mag2(data[0]);
    real_4 mini = mag2(data[0]);
    for (int ii=0; ii<length; ii++) {
      real_4 ab = mag2(data[ii]);
      if (ab>maxi) maxi = ab;
      if (ab<mini) mini = ab;
    }
     
    // Post to attributes
    TransactionLock tl(attrs);
    attrs["Max"] = maxi;
    attrs["Min"] = mini;
  }

}; // MaxMin

class Empty : public Analyzer {
public:
  Empty (const string& name) : Analyzer(name) { }

  // Do nothing
  virtual void analyze (const Val&)
  { }

}; // Empty


