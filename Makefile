CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp -march=native
HDF5_FLAGS = $(shell pkg-config --cflags --libs hdf5)

TARGET = read_h5
SRC = read_h5.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(HDF5_FLAGS) -lhdf5_cpp

clean:
	rm -f $(TARGET)

.PHONY: all clean