# Find the Torch includes and library
#
# Torch_FOUND -- set to 1 if found
SET(Torch_FOUND 1)
get_filename_component(TH_DIRECTORY "torch/torch7/lib/TH" REALPATH BASE_DIR ${CMAKE_CURRENT_LIST_DIR}/../)
MESSAGE(STATUS "TH directory: ${TH_DIRECTORY}")
set(THNN_INSTALL_LIB_SUBDIR ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(Torch_INSTALL_LIB ${TH_DIRECTORY}/build)
INCLUDE_DIRECTORIES(${TH_DIRECTORY} ${TH_DIRECTORY}/build)

IF(CUDA)
    get_filename_component(THC_DIRECTORY "cutorch/cutorch/lib/THC" REALPATH BASE_DIR ${CMAKE_CURRENT_LIST_DIR}/../)
    MESSAGE(STATUS "THC directory: ${THC_DIRECTORY}")
    INCLUDE_DIRECTORIES(${THC_DIRECTORY}/.. ${THC_DIRECTORY} ${THC_DIRECTORY}/build)
ENDIF(CUDA)
