TARGETS=Train Detect
SOURCES=Train.cpp Detect.cpp 
OBJECTS=common.o adaboost.o haarlike.o
HEADER=common.h adaboost.h haarlike.h

CXX = g++
CXXFLAGS = `pkg-config --cflags opencv` -O3 -Wall -I. -fopenmp
LDFLAGS = `pkg-config --libs opencv` -lgomp

all: $(OBJECTS) $(TARGETS)

$(OBJECTS): $(OBJECTS:%.o=%.h) $(SOURCES)

$(TARGETS): $(OBJECTS) 
	$(CXX) $(OBJECTS) $@.cpp -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -vf $(OBJS) $(TARGETS) *.o *~
