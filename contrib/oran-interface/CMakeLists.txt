# Include directories for e2sim
include_directories(/usr/local/include/e2sim)
# Linker flags and libraries
link_directories(/usr/local/lib)
link_libraries(e2sim)

# Add ns3 include directory from build folder
include_directories(${CMAKE_BINARY_DIR}/include/ns3)

# Define the source files
set(source_files
    model/oran-interface.cc
    model/asn1c-types.cc
    model/function-description.cc
    model/kpm-indication.cc
    model/kpm-function-description.cc
    model/ric-control-message.cc
    model/ric-control-function-description.cc
    helper/oran-interface-helper.cc
    helper/indication-message-helper.cc
    helper/lte-indication-message-helper.cc
    helper/mmwave-indication-message-helper.cc
)

# Define the header files
set(header_files
    model/oran-interface.h
    model/asn1c-types.h
    model/function-description.h
    model/kpm-indication.h
    model/kpm-function-description.h
    model/ric-control-message.h
    model/ric-control-function-description.h
    helper/oran-interface-helper.h
    helper/indication-message-helper.h
    helper/lte-indication-message-helper.h
    helper/mmwave-indication-message-helper.h
)

# Define the test sources (empty if no tests are needed)
set(test_sources "")  # Initialize test_sources to prevent CMake warnings

# Create the oran-interface library using build_lib
build_lib(
  LIBNAME oran-interface
  SOURCE_FILES ${source_files}
  HEADER_FILES ${header_files}
  LIBRARIES_TO_LINK
    ${libcore}
  e2sim
  TEST_SOURCES ${test_sources}
)

# Specify the install directory for ns3 includes
set(NS3_INSTALL_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/ns3)

# Install headers to the correct location
install(FILES
    ${header_files}
    DESTINATION ${NS3_INSTALL_INCLUDE_DIR}
)
