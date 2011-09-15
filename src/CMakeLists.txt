add_library(mmod SHARED
    mmod_feature.cpp
    mmod_general.cpp
    mmod_mode.cpp
    mmod_objects.cpp
    mmod_color.cpp
    )

target_link_libraries(mmod ${OpenCV_LIBS} boost_serialization)

add_executable(MyMMod MyMMod.cpp)
target_link_libraries(MyMMod mmod ${OpenCV_LIBS} boost_serialization)
