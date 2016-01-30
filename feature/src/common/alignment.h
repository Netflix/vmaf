#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

// int vmaf_floorn(int n, int m) // O0
inline int vmaf_floorn(int n, int m) // O1, O2, O3
{
	return n - n % m;
}

// int vmaf_ceiln(int n, int m) // O0
inline int vmaf_ceiln(int n, int m) // O1, O2, O3
{
	return n % m ? n + (m - n % m) : n;
}

#endif // ALIGNMENT_H_
