

// ///////////////////////////////////////////// Include Files

#include "occontainer_test.h"
#include "ocavltree.h"      

#if defined(OC_FORCE_NAMESPACE)
 using namespace OC;
#endif

template <class K, class V, int_u4 N>
ostream& operator<< (ostream& os, const AVLTreeT<K,V,N>& oo)
{
  for (AVLTreeTIterator<K,V,N> ii(oo); ii(); ) {
    os << ii.key() << " " << ii.value() << endl;
  }
  return os;
}

// ///////////////////////////////////////////// Main Program

int main ()
{

  {
    AVLTree<string> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<1; ii++) {
      a.insertKeyAndValue(Stringize(ii), Stringize(ii));
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) 
	cout << "OKAY" << endl; 
      else cout << "UH-oh!" << endl;
    // a.infix(cout);
    }
  }

  {
    AVLTree<string> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=9; ii!=-1; ii--) {
      a.insertKeyAndValue(Stringize(ii), Stringize(ii));
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      // a.infix(cout);
    }
  }

  {
    cout << "Trigger a double-rotate" << endl;
    int nums[] = { 4, 2, 3 };
    AVLTree<string> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<3; ii++) {
      a.insertKeyAndValue(Stringize(nums[ii]), Stringize(nums[ii]));
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      // a.infix(cout);
    }
  }

  {
    cout << "Trigger a more complex double-rotate" << endl;
    real_4 nums[] = { 10, 5, 15, 3, 7, 12, 20, 2, 4, 6, 8 };
    AVLTreeT<real_4, real_4, 8> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<11; ii++) {
      a.insertKeyAndValue(nums[ii], nums[ii]);
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      // a.infix(cout);
    }
    a.insertKeyAndValue(7.5, 7.5);
    a.prefix(cout); cout << endl;
    a.infix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
  }

  {
    cout << "Trigger a more complex double-rotate" << endl;
    real_4 nums[] = { 10, 5, 15, 3, 7, 12, 20, 2, 4, 6, 8 };
    AVLTreeT<real_4, real_4, 8> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<11; ii++) {
      a.insertKeyAndValue(nums[ii], nums[ii]);
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      // a.infix(cout);
    }
    a.insertKeyAndValue(6.5, 6.5);
    a.prefix(cout); cout << endl;
    a.infix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
  }


  {
    cout << "Trigger a more complex double-rotate" << endl;
    real_4 nums[] = { 10, 5, 15, 3, 7, 12, 20, 11, 13, 16, 25 };
    AVLTreeT<real_4, real_4,8> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<11; ii++) {
      a.insertKeyAndValue(nums[ii], nums[ii]);
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      // a.infix(cout);
    }
    a.insertKeyAndValue(13.5, 13.5);
    a.prefix(cout); cout << endl;
    a.infix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
  }

  {
    cout << "Trigger a more complex double-rotate" << endl;
    real_4 nums[] = { 10, 5, 15, 3, 7, 12, 20, 11, 13, 16, 25 };
    AVLTreeT<real_4, real_4,8> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<11; ii++) {
      a.insertKeyAndValue(nums[ii], nums[ii]);
      a.prefix(cout); cout << endl;
      a.infix(cout);
      if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      // a.infix(cout);
    }
    a.insertKeyAndValue(12.5, 12.5);
    a.prefix(cout); cout << endl;
    a.infix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
  }


  {
    cout << "Random!" << endl;
    AVLTreeT<real_4, real_4, 8> a;
    
    a.prefix(cout);
    if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
    for (int ii=0; ii<80; ii++) {
      //for (int ii=0; ii<10000; ii++) {
      // srand(M2Time().seconds());
      real_4 num = rand();
      // cout << "Inserting: " << num << endl;
      a.insertKeyAndValue(num, num);
      // a.prefix(cout); cout << endl;
      // a.infix(cout);
      if (!a.consistent()) { cout << "UH-oh!"; exit(1); }
      // if (ii%100==0) 
	cout << ".";
      //cout << endl << "Rec. Infix: ";
      // a.infix(cout);
    }
  }


  {
    cout << "Deletions" << endl;

    
    for (int ii=0; ii<10; ii++) {
      AVLTreeT<int_4, int_4, 8> a;
      for (int jj=0; jj<ii; jj++) {
	a.insertKeyAndValue(jj, jj);
      }
      cout << "Here we go:";
      a.infix(cout); cout << endl;
      for (int kk=0; kk<ii; kk++) {
	cout << "trying to delete: " << kk << endl;
	a.remove(kk);
	a.prefix(cout);
	if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      }
      cout << " ... result of delete: "; a.infix(cout); cout << endl;
    }
    
  }


  {
    cout << "Other way Deletions" << endl;

    
    for (int ii=9; ii>0; ii--) {
      AVLTreeT<int_4, int_4, 8> a;
      for (int jj=ii; jj>0; jj--) {
	a.insertKeyAndValue(jj, jj);
      }
      cout << "Here we go:";
      a.infix(cout); cout << endl;
      for (int kk=ii; kk>0; kk--) {
	cout << "trying to delete: " << kk << endl;
	a.remove(kk);
	a.prefix(cout);
	if (a.consistent()) cout << "OKAY" << endl; else cout << "UH-oh!" << endl;
      }
      cout << " ... result of delete: "; a.infix(cout); cout << endl;
    }
    
  }

  {
    AVLTreeT<int_4, int_4, 8> a;
    a.insertKeyAndValue(2,2);
    a.insertKeyAndValue(1,1);
    a.prefix(cout); cout << endl;
    a.remove(2);
    a.prefix(cout); cout << endl;
    a.remove(1);
    a.prefix(cout); cout << endl;
  }

#define MAGIC_NUM 100
  {
    for (int kk=0; kk<2; kk++) {
      cout << "TIME: " << kk << endl;
      
      AVLTreeT<int,int,8> a;
      
      int nums[MAGIC_NUM];
      
      for (int ii=0; ii<MAGIC_NUM; ii++) {
	nums[ii] = ii;
	a.insertKeyAndValue(ii, ii);
      }

      for (int jj=MAGIC_NUM-1; jj>0; jj--) {
	int ind = abs(rand())%jj;
	int temp = nums[ind];
	nums[ind] = nums[jj];
	nums[jj] = temp;
      }
      a.infix(cout); cout << endl;
      for (int ll=0; ll<MAGIC_NUM; ll++) {
	// cout << nums[ll] << " ";
	a.remove(nums[ll]);
	if (!a.consistent()) { cout << "UH_OH!" << endl; }
      }
    }
  }


  ContainerTest<AVLTree<int_u4>, AVLTreeIterator<int_u4>,
    AVLTree<string>, AVLTreeIterator<string> > t;
  (void)t.tests();


  // test to make sure you can insert into a table "quicker" with
  // swapInto
  AVLTreeT<string, string, 8> oo;
  for (int ii=0; ii<1000; ii++) {
    string key = Stringize(ii);
    string value = Stringize(ii) + "The quick brown fox jumped over the lazy dogs enough times to make a string that exceeded the lookaside cache";
    oo.swapInto(key, value);
  }
  for (AVLTreeTIterator<string, string,8> ii(oo); ii(); ) {
    cout << ii.key() << " " << ii.value() << endl;
  }

  // make sure you can compare Trees too!
  cout << "** Make sure you can compare Trees" << endl;
  {
    AVLTreeT<char, int, 8> t1, t2;
    t1['a'] = 1;   t1['b'] = 2; t1['c'] = 3;
    t2['a'] = 1;   t2['b'] = 2; t2['c'] = 3;
    cout << t1 << "\n" << t2 << endl;
    cout << bool(t1==t2) << endl;
    cout << bool(t1!=t2) << endl;
    cout << bool(t1<t2) << endl;
    cout << bool(t1<=t2) << endl;
    cout << bool(t1>=t2) << endl;
    cout << bool(t1>t2) << endl;
  }

  {
    AVLTreeT<char, int, 8> t1, t2;
    t1['a'] = 1;   t1['b'] = 2; t1['c'] = 3;
    t2['a'] = 1;   t2['c'] = 2; 
    cout << t1 << "\n" << t2 << endl;
    cout << bool(t1==t2) << endl;
    cout << bool(t1!=t2) << endl;
    cout << bool(t1<t2) << endl;
    cout << bool(t1<=t2) << endl;
    cout << bool(t1>=t2) << endl;
    cout << bool(t1>t2) << endl;
  }

  {
    AVLTreeT<char, int, 8> t1, t2;
    t1['a'] = 1;   t1['b'] = 2; t1['d'] = 3;
    t2['a'] = 1;   t2['c'] = 3; t2['d'] = 3;
    cout << t1 << "\n" << t2 << endl;
    cout << bool(t1==t2) << endl;
    cout << bool(t1!=t2) << endl;
    cout << bool(t1<t2) << endl;
    cout << bool(t1<=t2) << endl;
    cout << bool(t1>=t2) << endl;
    cout << bool(t1>t2) << endl;
  }
  
  {
    AVLTreeT<char, int, 8> t1, t2;
    t1['a'] = 1;   t1['b'] = 2; t1['c'] = 3;
    t2['a'] = 1;   t2['b'] = 3; t2['c'] = 3;
    cout << t1 << "\n" << t2 << endl;
    cout << bool(t1==t2) << endl;
    cout << bool(t1!=t2) << endl;
    cout << bool(t1<t2) << endl;
    cout << bool(t1<=t2) << endl;
    cout << bool(t1>=t2) << endl;
    cout << bool(t1>t2) << endl;
  }

  
}



