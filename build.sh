rm build -rf
mkdir build && cd build

# 关键：不使用系统库，只用本地 third_party
cmake .. \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)