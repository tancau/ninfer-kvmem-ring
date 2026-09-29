add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)
# LOCAL DIAG (crash-symbols): dbghelp turns the SIGABRT handler's raw frames into names. The PDB has
# to sit next to the deployed exe for SymInitialize to find it.
if(NINFER_CRASH_SYMBOLS)
  target_link_libraries(ninfer-serve PRIVATE dbghelp)
  target_compile_definitions(ninfer-serve PRIVATE NINFER_CRASH_SYMBOLS=1)
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
