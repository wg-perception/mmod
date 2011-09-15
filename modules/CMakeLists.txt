find_package(ecto)
if(NOT ECTO_FOUND)
    message(STATUS "Ecto was not found, so disabling ecto cells")
endif()

find_package(OpenCV REQUIRED )

ecto_python_env_gen(${CMAKE_BINARY_DIR}/lib)

ectomodule(mmod
 module.cpp
 MModTrainer.cpp
 MModPersister.cpp
)

link_ecto(mmod
    mmod
    ${OpenCV_LIBS} 
)
