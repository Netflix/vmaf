
// ///////////////////////////////////////////// HashFunction
#include "ocport.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// In newer Linux GNU C++ compilers, HashFunctions have to come before include?

// For Koenig lookup to find these HashFunctions, they have to be
// thrown into the namespace
OC_BEGIN_NAMESPACE

inline int_u4 HashFunction (const real_8& d)
{
  return (int_u4) (d*10);
}

inline int_u4 HashFunction (const int_4& d)
{
  return (int_u4) d;
}

inline int_u4 HashFunction (const real_4& d)
{
  return (int_u4) (d*10);
}

OC_END_NAMESPACE

// ///////////////////////////////////////////// Include Files

#include "occontainer_test.h"
#include "ocavlhash.h"      


// ///////////////////////////////////////////// Main Program

int main ()
{
  {
    AVLHash<string> a;
    
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
    AVLHash<string> a;
    
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
    cout << "More inserts..." << endl;
    int nums[] = { 4, 2, 3 };
    AVLHash<string> a;
    
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
    cout << "Trigger complex double rotate..." << endl;
    real_4 nums[] = { 10, 5, 15, 3, 7, 12, 20, 2, 4, 6, 8 };
    AVLHashT<real_4, real_4, 8> a;
    
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
    AVLHashT<real_4, real_4, 8> a;
    
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
    AVLHashT<real_4, real_4,8> a;
    
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
    AVLHashT<real_4, real_4,8> a;
    
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
    AVLHashT<real_4, real_4, 8> a;
    
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
      AVLHashT<int_4, int_4, 8> a;
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
      AVLHashT<int_4, int_4, 8> a;
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
    AVLHashT<int_4, int_4, 8> a;
    a.insertKeyAndValue(2,2);
    a.insertKeyAndValue(1,1);
    a.prefix(cout); cout << endl;
    a.remove(2);
    a.prefix(cout); cout << endl;
    a.remove(1);
    a.prefix(cout); cout << endl;
  }

#define MAGIC_NUM 1000
  {
    for (int kk=0; kk<4; kk++) {
      cout << "TIME: " << kk << endl;
      
      AVLHash<int> a;
      
      int nums[MAGIC_NUM];
      
      for (int ii=0; ii<MAGIC_NUM; ii++) {
	nums[ii] = ii;
	a.insertKeyAndValue(Stringize(ii), ii);
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
	a.remove(Stringize(nums[ll]));
	if (!a.consistent()) { cout << "UH_OH!" << endl; }
      }
    }
  }
  


  ContainerTest<AVLHash<int_u4>, AVLHashIterator<int_u4>,
    AVLHash<string>, AVLHashIterator<string> > t;
  return t.tests();
}



