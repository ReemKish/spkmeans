CFLAGS += -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS += -lm

all: spkmeans module

spkmeans: spkmeans.o
	$(CC) $(CFLAGS) spkmeans.o -o spkmeans $(LDFLAGS)

module: spkmeansmodule.c spkmeans.o setup.py
	python3 setup.py build_ext --inplace

spkmeans.o: spkmeans.c spkmeans.h
	$(CC) $(CFLAGS) -c spkmeans.c
spkmeanssp: spkmeans.c setup.py
	python3 setup.py build_ext --inplace
clean:
	rm -rfd spkmeans *.so *.o build
