mkdir build_raspberry
cd build_raspberry
cmake -DARM7=TRUE ..
make

cp ../model/* .

