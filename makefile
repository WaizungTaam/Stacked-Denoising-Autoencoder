CC=g++
FLAG=-std=c++11
PATH_SRC=./src/
PATH_MATH=./src/math/
PATH_TEST=./test/
SDA=$(PATH_SRC)sdA.h $(PATH_SRC)sdA.cc
DA=$(PATH_SRC)dA.h $(PATH_SRC)dA.cc
MLP=$(PATH_SRC)mlp.h $(PATH_SRC)mlp.cc
MATH=$(PATH_MATH)*

all: mlp_test.o dA_test.o sdA_test.o

sdA_test.o: $(MATH) $(DA) $(MLP) $(SDA) $(PATH_TEST)sdA_test.cc
	$(CC) $(FLAG) $(MATH) $(DA) $(MLP) $(SDA) $(PATH_TEST)sdA_test.cc -o sdA_test.o

dA_test.o: $(MATH) $(DA) $(PATH_TEST)dA_test.cc
	$(CC) $(FLAG) $(MATH) $(DA) $(PATH_TEST)dA_test.cc -o dA_test.o

mlp_test.o: $(MATH) $(MLP) $(PATH_TEST)mlp_test.cc
	$(CC) $(FLAG) $(MATH) $(MLP) $(PATH_TEST)mlp_test.cc -o mlp_test.o

clean:
	rm *.o