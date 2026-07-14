CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -g
TARGET   := tensorgrad
SRCS     := main.cpp
HEADERS  := matrix.hpp value.hpp

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
