#include "3rd_party/catch.hpp"

#include "config.h"

#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while(0)
#define CHECK_FALSE_MESSAGE(cond, msg) do { INFO(msg); CHECK_FALSE(cond); } while(0)
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while(0)
#define REQUIRE_FALSE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE_FALSE(cond); } while(0)

#define CHECK_EPS(actual, expected, epsilon) \
  do { \
    if (fabs((actual) - (expected)) < epsilon) { SUCCEED(); } \
    else { CHECK((actual) == (expected)); } \
  } while(0)

#define KERNEL_TEST_CASE(name) TEST_CASE("Kernel: " name, "[kernel_test]")
