#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        std::cout << "Hello from thread "
                  << omp_get_thread_num()
                  << " of "
                  << omp_get_num_threads()
                  << '\n';
    }

    return 0;
}