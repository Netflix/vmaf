// ///////////////////////////////////////////// Hashing

OC_BEGIN_NAMESPACE

// HashFunction stolen directly out of Python 2.7: This is a very good hash
// function and much simpler and smaller footprint than Pearson's.
inline unsigned long OCStringHashFunction (const char* c_data, int length)
{
  int len = length;
  register long x = *c_data<<7;
  while (--len >=0) 
    x =  (1000003*x) ^ *c_data++;
  x ^= length;
  return x;
}


OC_END_NAMESPACE

