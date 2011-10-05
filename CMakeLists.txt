cmake_minimum_required(VERSION 2.8)
#set(CMAKE_BUILD_TYPE Release)
project(mmod)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS serialization REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
    )
add_subdirectory(src)

