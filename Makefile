CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
TARGET = main
SRCS = main.cpp Network.cpp Layer.cpp mnist_reader.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

