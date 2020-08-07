#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <iterator>
#include <chrono>
#include <type_traits>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "algorithms/algorithm.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/count.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"
#include "proto/data.pb.h"
#include "base/statusor.h"
#include "algorithms/count.h"

#include "testing/sequence.h"
#include "absl/memory/memory.h"

#include "testing/stochastic_tester.h"

namespace differential_privacy {

namespace testing {

template <typename T,
              typename std::enable_if<std::is_integral<T>::value ||
                                      std::is_floating_point<T>::value>::type* =
                  nullptr>

class InsufficientNoiseSum : public differential_privacy::BoundedSum<T> {
     public:
      InsufficientNoiseSum(double ratio,
          double epsilon, T lower, T upper,
          std::unique_ptr<LaplaceMechanism::Builder> builder) // getting a builder from a caller 
          : BoundedSum<T>(epsilon, lower, upper, 1, 1, std::move(builder), nullptr, nullptr), ratio_(ratio) {} // setting sensitivity values to 1
      double GetEpsilon() const override { return Algorithm<T>::GetEpsilon() * ratio_; }
    private: 
      double ratio_;
  };

  bool StochasticTest_Sum(double ratio, double num_datasets, double num_samples_per_histogram) {

    // Create Halton sequence
    auto sequence = absl::make_unique<differential_privacy::testing::HaltonSequence<double>>(
      DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());
    
    auto algorithm = absl::make_unique<InsufficientNoiseSum<double>>(ratio,DefaultEpsilon(),sequence->RangeMin(), sequence->RangeMax(),
      absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>());
    
    StochasticTester<double> tester(
        std::move(algorithm), std::move(sequence), num_datasets, num_samples_per_histogram);
    
    bool algo_is_dp = tester.Run();
    
    return algo_is_dp;

  }

bool GetTestResult(double ratio, double num_datasets, double num_samples_per_histogram) {

  bool algo_is_dp = StochasticTest_Sum(ratio, num_datasets, num_samples_per_histogram);

  std::cout << algo_is_dp << std::endl;

  return algo_is_dp; // false = 0, true = 1

}

void GetTestPerformanceSummary(std::ofstream& datafile, std::ofstream& summaryfile, double num_datasets, double num_samples_per_histogram) {

  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio = 0;
  auto start_loop = std::chrono::high_resolution_clock::now();

  for (double i=85.0; i<=99.0; i++) {

    auto start = std::chrono::high_resolution_clock::now();

    double ratio = i/100.0;

    std::cout << "Now calculating algorithm with ratio of " << ratio << std::endl;

    bool outcome = GetTestResult(ratio, num_datasets, num_samples_per_histogram);

    num_tests = num_tests+1;

    if (outcome == 0) { // Correct outcome is false, e.g., DP predicate rejected 

      num_tests_passed = num_tests_passed+1;

      maximum_ratio = ratio;

    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);

    datafile << "insufficient_noise,bounded_sum,0," << outcome << "," << ratio << "," << num_datasets << "," << num_samples_per_histogram << "," << elapsed.count() << "\n";

  }

    double accuracy = num_tests_passed/num_tests;

    auto finish_loop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_loop = std::chrono::duration_cast<std::chrono::duration<double>>(finish_loop - start_loop);

    summaryfile << "insufficient_noise,bounded_sum," << num_tests << "," << num_tests_passed << "," << accuracy << "," << maximum_ratio << "," << num_datasets << "," << num_samples_per_histogram << "," << elapsed_loop.count() << "\n";

    std::cout << num_tests_passed << " out of " << num_tests << " passed." << std::endl;
}
}
}

int main(int argc, char** argv) {

  std::ofstream datafile;
  datafile.open("/usr/local/google/home/krosman/myproject/stochastic_tester_results_boundedsum.txt");
  datafile << "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)" << "\n";

  std::ofstream summaryfile;
  summaryfile.open("/usr/local/google/home/krosman/myproject/stochastic_tester_results_summary_boundedsum.txt");
  summaryfile << "test_name,algorithm,num_tests,num_successful,accuracy,maximum_ratio,num_datasets,num_samples,time(sec)" << "\n";

  differential_privacy::testing::GetTestPerformanceSummary(datafile,summaryfile,15,1000000);

  datafile.close();
  summaryfile.close();

  return 0;
}