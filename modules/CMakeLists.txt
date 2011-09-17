find_package(ecto)
#set(CMAKE_BUILD_TYPE Release)
if(NOT ecto_FOUND)
  message(STATUS "Ecto was not found, so disabling ecto cells")
  return()
endif()

find_package(OpenCV REQUIRED )

ecto_python_env_gen(${CMAKE_BINARY_DIR}/lib)

ectomodule(mmod
 module.cpp
 MModTrainer.cpp
 MModTester.cpp
 MModPersister.cpp
)

link_ecto(mmod
    mmod
    ${OpenCV_LIBS} 
)
