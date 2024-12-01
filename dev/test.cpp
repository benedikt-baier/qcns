#include <iostream>

#include <chrono>

#include <Eigen/Dense>

int main() {
    const int size = 10;
    Eigen::MatrixXd mat1 = Eigen::MatrixXd::Random(size, size);
    Eigen::MatrixXd mat2 = Eigen::MatrixXd::Random(size, size);
    Eigen::MatrixXcd cmat1 = Eigen::MatrixXcd::Random(size, size);
    Eigen::MatrixXcd cmat2 = Eigen::MatrixXcd::Random(size, size);

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd result1 = mat1 + mat2;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time for real matrix addition: " << elapsed.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXcd result2 = cmat1 + cmat2;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time for complex matrix addition: " << elapsed.count() << " seconds\n";

    return 0;
}