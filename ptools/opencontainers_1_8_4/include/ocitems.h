#ifndef OCITEMS_H_
#define OCITEMS_H_

// Helper functions for getting the keys and values out of Tab and OTab.
// NOTE we don't include anything here, as we assume this gets included by ocval.cc
// at the proper place

OC_BEGIN_NAMESPACE

template <class CON>
inline Arr keys (const CON& container)
{
  Arr res(container.entries());
  It ii(container);
  while (ii()) {
    const Val& key = ii.key();
    res.append(key);
  }
  return res;
}

template <class CON>
inline Arr values (const CON& container)
{
  Arr res(container.entries());
  It ii(container);
  while (ii()) {
    Val& value = ii.value();
    res.append(value);
  }
  return res;
}

template <class CON>
inline Arr items (const CON& container)
{
  Arr res(container.entries());
  It ii(container);
  while (ii()) {
    const Val& key = ii.key();
    Val& value = ii.value();
    Tup t(key, value);
    res.append(t);
  }
  return res;
}

OC_END_NAMESPACE

#endif // OCITEMS_H_
