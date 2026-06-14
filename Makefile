CXX = g++
CXXFLAGS = -std=c++17 -I/opt/homebrew/include/eigen3 -Wall -O2 	# where u installed eigen
TARGET = main
SRCS = main.cpp mnist_reader.cpp Network.cpp Layer.cpp
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

