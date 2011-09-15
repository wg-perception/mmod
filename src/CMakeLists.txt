find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(mmod mmod_feature.cpp mmod_general.cpp mmod_mode.cpp mmod_objects.cpp mmod_color.cpp MyMMod.cpp)

target_link_libraries(mmod ${OpenCV_LIBS} ${Boost_LIBRARIES})
