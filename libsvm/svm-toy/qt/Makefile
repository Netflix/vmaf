CXX? = g++
CFLAGS = -Wall -O3 -I$(INCLUDE) -I$(INCLUDE)/QtGui -lQtGui
INCLUDE = /usr/include/qt4
MOC = /usr/bin/moc-qt4

svm-toy: svm-toy.cpp svm-toy.moc ../../svm.o
	$(CXX) $(CFLAGS) svm-toy.cpp ../../svm.o -o svm-toy

svm-toy.moc: svm-toy.cpp
	$(MOC) svm-toy.cpp -o svm-toy.moc

../../svm.o: ../../svm.cpp ../../svm.h
	make -C ../.. svm.o

clean:
	rm -f *~ svm-toy svm-toy.moc ../../svm.o

