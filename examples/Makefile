CC = $(CXX)

CXXFLAGS += -O3 -Wall -Wextra -pedantic -std=c++14
LIBFLAGS = -fPIC
EXEC = example
RM = rm -f

INCLUDES_DIRS = ./ ../include
INCLUDES = $(shell find $(INCLUDES_DIRS) -type f -name '*.h')
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean run cppcheck lint

all: $(EXEC)
$(EXEC): $(OBJECTS)
$(OBJECTS): $(SOURCES)

clean:
	$(RM) *~ $(EXEC) $(OBJECTS)

run: $(EXEC)
	./$(EXEC)

cppcheck:
	cppcheck -v --std=c++11 --language=c++ --platform=native --enable=all --inconclusive --inline-suppr $(INCLUDES) $(SOURCES)

lint:
	cpplint $(INCLUDES) $(SOURCES)

