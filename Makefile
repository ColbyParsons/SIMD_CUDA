SOURCES = matrixMult.cc
HEADERS =

OBJECTS = $(SOURCES:%.cc=%.o)
PROGRAM = matrixMult

CC := g++
CFLAGS = -msse4.1 -mavx -mavx2 -std=c++17
WARNINGS = -Wall -Wextra
LDFLAGS =

$(PROGRAM) : $(OBJECTS)
		$(CC) $(LDFLAGS) -o $@ $<

%.o : %.cc $(HEADERS)
		$(CC) $(CFLAGS) -c -o $@ $<

.PHONY : clean
clean :
		rm -f $(PROGRAM) $(OBJECTS)