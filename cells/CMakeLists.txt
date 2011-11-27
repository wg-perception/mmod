find_package(ecto)
if(NOT ecto_FOUND)
  return()
endif()

ecto_python_env_gen(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

include_directories(../../object_recognition/include)
include_directories(../src)

ectomodule(mmod
   module.cpp
   MModModelFiller.cpp
   MModTester.cpp
   MModTrainer.cpp
   MModPersister.cpp
   Pyramid.cpp
)

link_ecto(mmod
  mmod
  object_recognition_db
  ${OpenCV_LIBS}
)

