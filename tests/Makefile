CC = $(CXX)

CXXFLAGS += -O3 -Wall -Wextra -pedantic -std=c++14
GCOV_FLAGS = --coverage
LDLIBS = -lgtest -lpthread
LIBFLAGS = -fPIC
EXEC = ddsketch_tests
RM = rm -f

INCLUDES_DIRS = ./ ../include
INCLUDES = $(shell find $(INCLUDES_DIRS) -type f -name '*.h')
SOURCES = ddsketch_tests.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean run cppcheck lint

all: $(EXEC)
$(EXEC): $(OBJECTS)
$(OBJECTS): $(SOURCES)

clean:
	$(RM) *~ *.gcda *.gcno $(EXEC) $(OBJECTS)

run: $(EXEC)
	./$(EXEC)

coverage: $(SOURCES)
	$(CC) $(CXXFLAGS) $(GCOV_FLAGS) $(SOURCES) $(LDLIBS) -o $(EXEC)
	./$(EXEC)

cppcheck:
	cppcheck -v --std=c++11 --language=c++ --platform=native --enable=all --inconclusive --inline-suppr $(INCLUDES) $(SOURCES)

lint:
	cpplint $(INCLUDES) $(SOURCES)

