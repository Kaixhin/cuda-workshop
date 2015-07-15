#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

int my_rand() {
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9999);
  return dist(rng);
}

int main() {
  // Generate random data on host
  thrust::host_vector<int> h_vec(100);
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

  thrust::device_vector<int> d_vec = h_vec; // Transfer to device

  int init = 0; // Initial value of reduction

  thrust::plus<int> binary_op; // Binary operation used to reduce values

  int sum = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op); // Compute sum on device

  std::cout << "Sum is " << sum << std::endl; // Print sum

  return 0;
}
