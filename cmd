cmake -S datasets/sqlitebrowser -B build/sqlitebrowser \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DQT_MAJOR=Qt6