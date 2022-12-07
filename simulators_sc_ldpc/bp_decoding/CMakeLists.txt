cmake_minimum_required(VERSION 3.14)
project(bp C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -Wpedantic")

add_executable(bp_lim_iter SC_LDPC_Simulator_BPDecoder_BEC_full_BP_LimIter_OlmosRandomEnsemble.c)
add_executable(sw_lim_iter SC_LDPC_Simulator_BPDecoder_BEC_SlidingWindow_LimIter_OlmosRandomEnsemble.c)
add_executable(bp_traj trajectories_SC_LDPC_Simulator_BPDecoder_BEC_full_BP_OlmosRandomEnsemble.c)

target_link_libraries(bp_lim_iter m)
target_link_libraries(sw_lim_iter m)
target_link_libraries(bp_traj m)

#find_package(GSL REQUIRED)
#target_link_libraries(mevol GSL::gsl)
