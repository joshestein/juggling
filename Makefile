CC=g++ -std=c++11
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECS=$(SOURCES:.cpp=)
FLAGS=`pkg-config --libs --cflags opencv`

all:
	$(CC) $(SOURCES) -o $(EXECS) $(FLAGS)
# all: $(OBJECTS) $(EXECS)

clean:
	rm -rf $(OBJECTS) $(EXECS)

$(OBJECTS): %.o: %.cpp

$(EXECS): %: %.o
	$(CC) $(FLAGS) $< -o $@

run:
	./main
