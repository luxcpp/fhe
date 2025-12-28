//==================================================================================
// GPU TFHE Unit Tests and Benchmarks
// Target: 1000+ concurrent operations, massive parallelism
//==================================================================================

#include "gtest/gtest.h"
#include "math/hal/mlx/gpu_tfhe.h"
#include <chrono>
#include <random>
#include <thread>
#include <atomic>

using namespace lbcrypto::gpu;
using namespace std::chrono;

class GPUTFHETest : public ::testing::Test {
protected:
    void SetUp() override {
        TFHEConfig config;
        config.N = 1024;
        config.n = 512;
        config.L = 4;  // Reduced from 7!
        config.maxUsers = 1000;
        config.batchSize = 256;
        
        engine_ = std::make_unique<GPUTFHEEngine>(config);
        ASSERT_TRUE(engine_->initialize());
    }
    
    void TearDown() override {
        engine_->shutdown();
    }
    
    std::unique_ptr<GPUTFHEEngine> engine_;
};

//==================================================================================
// Basic Tests
//==================================================================================

TEST_F(GPUTFHETest, EngineInitialization) {
    EXPECT_EQ(engine_->activeUsers(), 0);
    EXPECT_GT(engine_->availableGPUMemory(), 0);
}

TEST_F(GPUTFHETest, UserCreation) {
    uint64_t user1 = engine_->createUser();
    uint64_t user2 = engine_->createUser();
    
    EXPECT_NE(user1, user2);
    EXPECT_EQ(engine_->activeUsers(), 2);
    
    engine_->deleteUser(user1);
    EXPECT_EQ(engine_->activeUsers(), 1);
}

TEST_F(GPUTFHETest, CiphertextAllocation) {
    uint64_t userId = engine_->createUser();
    
    uint32_t poolIdx = engine_->allocateCiphertexts(userId, 100);
    EXPECT_EQ(poolIdx, 0);
    
    uint32_t poolIdx2 = engine_->allocateCiphertexts(userId, 200);
    EXPECT_EQ(poolIdx2, 1);
    
    EXPECT_GT(engine_->totalGPUMemoryUsed(), 0);
}

//==================================================================================
// Concurrency Tests
//==================================================================================

TEST_F(GPUTFHETest, ConcurrentUserCreation) {
    const int numUsers = 100;
    std::vector<std::thread> threads;
    std::vector<uint64_t> userIds(numUsers);
    std::atomic<int> successCount{0};
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < numUsers; ++i) {
        threads.emplace_back([this, i, &userIds, &successCount]() {
            try {
                userIds[i] = engine_->createUser();
                successCount++;
            } catch (...) {
                // Handle max user limit
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    std::cout << "Created " << successCount << " users in " << duration.count() << " ms" << std::endl;
    
    EXPECT_EQ(successCount, numUsers);
    EXPECT_EQ(engine_->activeUsers(), numUsers);
}

//==================================================================================
// Performance Benchmarks
//==================================================================================

TEST_F(GPUTFHETest, BenchmarkBatchNTT) {
    const std::vector<int> batchSizes = {1, 10, 100, 1000, 10000};
    
    std::cout << "\n=== Batch NTT Benchmark ===" << std::endl;
    std::cout << "Batch Size | Total Time (ms) | Per NTT (µs) | Throughput (NTT/s)" << std::endl;
    std::cout << "-----------|-----------------|--------------|-------------------" << std::endl;
    
    for (int batchSize : batchSizes) {
        // Create test data
        std::vector<int64_t> data(batchSize * 1024);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 27) - 1);
        
        for (auto& v : data) {
            v = static_cast<int64_t>(dist(gen));
        }
        
#ifdef WITH_MLX
        mx::array polys = mx::array(data.data(), {batchSize, 1024}, mx::int64);
        mx::eval(polys);
        
        // Warmup
        engine_->batchNTT(polys, false);
        engine_->sync();
        
        // Benchmark
        const int iterations = 10;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            engine_->batchNTT(polys, false);
        }
        engine_->sync();
        
        auto end = high_resolution_clock::now();
        double totalMs = duration_cast<microseconds>(end - start).count() / 1000.0;
        double perNTT = (totalMs * 1000.0) / (batchSize * iterations);
        double throughput = (batchSize * iterations * 1000.0) / totalMs;
        
        printf("%10d | %15.2f | %12.2f | %18.0f\n", 
               batchSize, totalMs, perNTT, throughput);
#endif
    }
}

TEST_F(GPUTFHETest, BenchmarkBatchExternalProduct) {
    const std::vector<int> batchSizes = {1, 10, 100, 500, 1000};
    
    std::cout << "\n=== Batch External Product Benchmark ===" << std::endl;
    std::cout << "Batch Size | Total Time (ms) | Per ExtProd (ms) | Throughput (ops/s)" << std::endl;
    std::cout << "-----------|-----------------|------------------|-------------------" << std::endl;
    
    for (int batchSize : batchSizes) {
#ifdef WITH_MLX
        // Create test RLWE and RGSW ciphertexts
        std::vector<int64_t> rlweData(batchSize * 2 * 1024);
        std::vector<int64_t> rgswData(batchSize * 2 * 4 * 2 * 1024);  // L=4
        
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 27) - 1);
        
        for (auto& v : rlweData) v = static_cast<int64_t>(dist(gen));
        for (auto& v : rgswData) v = static_cast<int64_t>(dist(gen));
        
        mx::array rlwe = mx::array(rlweData.data(), {batchSize, 2, 1024}, mx::int64);
        mx::array rgsw = mx::array(rgswData.data(), {batchSize, 2, 4, 2, 1024}, mx::int64);
        mx::array output;
        
        mx::eval(rlwe);
        mx::eval(rgsw);
        
        // Warmup
        engine_->batchExternalProduct(rlwe, rgsw, output);
        engine_->sync();
        
        // Benchmark
        const int iterations = 5;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            engine_->batchExternalProduct(rlwe, rgsw, output);
        }
        engine_->sync();
        
        auto end = high_resolution_clock::now();
        double totalMs = duration_cast<milliseconds>(end - start).count();
        double perOp = totalMs / (batchSize * iterations);
        double throughput = (batchSize * iterations * 1000.0) / totalMs;
        
        printf("%10d | %15.2f | %16.3f | %18.0f\n",
               batchSize, totalMs, perOp, throughput);
#endif
    }
}

TEST_F(GPUTFHETest, BenchmarkMassiveBatchGates) {
    const std::vector<int> numOps = {100, 1000, 5000, 10000};
    
    std::cout << "\n=== Massive Batch Gate Operations ===" << std::endl;
    std::cout << "Operations | Total Time (ms) | Per Gate (ms) | Throughput (gates/s)" << std::endl;
    std::cout << "-----------|-----------------|---------------|---------------------" << std::endl;
    
    // Create users with keys
    std::vector<uint64_t> userIds;
    for (int i = 0; i < 10; ++i) {
        userIds.push_back(engine_->createUser());
    }
    
    for (int ops : numOps) {
        // Create batch operations distributed across users
        BatchedGateOp batch;
        batch.gate = GateType::AND;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> idxDist(0, 999);
        std::uniform_int_distribution<size_t> userDist(0, userIds.size() - 1);
        
        for (int i = 0; i < ops; ++i) {
            batch.userIds.push_back(userIds[userDist(gen)]);
            batch.input1Indices.push_back(idxDist(gen));
            batch.input2Indices.push_back(idxDist(gen));
            batch.outputIndices.push_back(idxDist(gen));
        }
        
        auto start = high_resolution_clock::now();
        
        engine_->executeBatchGates({batch});
        engine_->sync();
        
        auto end = high_resolution_clock::now();
        double totalMs = duration_cast<milliseconds>(end - start).count();
        double perGate = totalMs / ops;
        double throughput = (ops * 1000.0) / totalMs;
        
        printf("%10d | %15.2f | %13.3f | %20.0f\n",
               ops, totalMs, perGate, throughput);
    }
}

TEST_F(GPUTFHETest, BenchmarkMultiUserParallel) {
    const int numUsers = 100;
    const int opsPerUser = 100;
    
    std::cout << "\n=== Multi-User Parallel Operations ===" << std::endl;
    std::cout << "Users: " << numUsers << ", Ops per user: " << opsPerUser << std::endl;
    
    // Create users
    std::vector<uint64_t> userIds;
    for (int i = 0; i < numUsers; ++i) {
        userIds.push_back(engine_->createUser());
        engine_->allocateCiphertexts(userIds.back(), 1000);
    }
    
    // Create operations for all users
    std::vector<BatchedGateOp> allOps;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> idxDist(0, 999);
    
    // AND gates
    BatchedGateOp andBatch;
    andBatch.gate = GateType::AND;
    
    // XOR gates
    BatchedGateOp xorBatch;
    xorBatch.gate = GateType::XOR;
    
    for (int u = 0; u < numUsers; ++u) {
        for (int i = 0; i < opsPerUser / 2; ++i) {
            andBatch.userIds.push_back(userIds[u]);
            andBatch.input1Indices.push_back(idxDist(gen));
            andBatch.input2Indices.push_back(idxDist(gen));
            andBatch.outputIndices.push_back(idxDist(gen));
            
            xorBatch.userIds.push_back(userIds[u]);
            xorBatch.input1Indices.push_back(idxDist(gen));
            xorBatch.input2Indices.push_back(idxDist(gen));
            xorBatch.outputIndices.push_back(idxDist(gen));
        }
    }
    
    allOps.push_back(std::move(andBatch));
    allOps.push_back(std::move(xorBatch));
    
    auto start = high_resolution_clock::now();
    
    engine_->executeBatchGates(allOps);
    engine_->sync();
    
    auto end = high_resolution_clock::now();
    double totalMs = duration_cast<milliseconds>(end - start).count();
    double totalOps = numUsers * opsPerUser;
    double throughput = (totalOps * 1000.0) / totalMs;
    
    std::cout << "Total time: " << totalMs << " ms" << std::endl;
    std::cout << "Total operations: " << totalOps << std::endl;
    std::cout << "Throughput: " << throughput << " ops/s" << std::endl;
    std::cout << "Per-operation time: " << (totalMs / totalOps) << " ms" << std::endl;
}

//==================================================================================
// Memory Pressure Test
//==================================================================================

TEST_F(GPUTFHETest, MemoryPressure) {
    std::cout << "\n=== Memory Pressure Test ===" << std::endl;
    
    size_t initialMem = engine_->totalGPUMemoryUsed();
    std::cout << "Initial GPU memory: " << (initialMem / (1024*1024)) << " MB" << std::endl;
    
    // Create many users with ciphertexts
    std::vector<uint64_t> users;
    const int numUsers = 50;
    const int ctsPerUser = 1000;
    
    for (int i = 0; i < numUsers; ++i) {
        uint64_t userId = engine_->createUser();
        users.push_back(userId);
        engine_->allocateCiphertexts(userId, ctsPerUser);
        
        if ((i + 1) % 10 == 0) {
            size_t currentMem = engine_->totalGPUMemoryUsed();
            std::cout << "After " << (i + 1) << " users: " 
                      << (currentMem / (1024*1024)) << " MB" << std::endl;
        }
    }
    
    size_t finalMem = engine_->totalGPUMemoryUsed();
    std::cout << "Final GPU memory: " << (finalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "Memory per user: " << ((finalMem - initialMem) / numUsers / (1024*1024)) << " MB" << std::endl;
    
    // Cleanup
    for (auto userId : users) {
        engine_->deleteUser(userId);
    }
    
    size_t afterCleanup = engine_->totalGPUMemoryUsed();
    std::cout << "After cleanup: " << (afterCleanup / (1024*1024)) << " MB" << std::endl;
}

//==================================================================================
// Scheduler Tests
//==================================================================================

TEST_F(GPUTFHETest, BatchScheduler) {
    uint64_t userId = engine_->createUser();
    engine_->allocateCiphertexts(userId, 1000);
    
    BatchPBSScheduler scheduler(engine_.get());
    scheduler.setAutoFlushThreshold(100);
    
    std::cout << "\n=== Batch Scheduler Test ===" << std::endl;
    
    auto start = high_resolution_clock::now();
    
    // Queue many operations
    for (int i = 0; i < 500; ++i) {
        scheduler.queueGate(userId, GateType::AND, i, i + 1, i + 500);
    }
    
    scheduler.flush();
    engine_->sync();
    
    auto end = high_resolution_clock::now();
    double totalMs = duration_cast<milliseconds>(end - start).count();
    
    std::cout << "Scheduled and executed 500 gates in " << totalMs << " ms" << std::endl;
}

//==================================================================================
// Circuit Evaluator Tests
//==================================================================================

TEST_F(GPUTFHETest, BatchedIntegerAdd) {
    uint64_t userId = engine_->createUser();
    engine_->allocateCiphertexts(userId, 10000);
    
    GPUCircuitEvaluator eval(engine_.get(), userId);
    
    std::cout << "\n=== Batched Integer Addition Test ===" << std::endl;
    
    // Create batch of 8-bit integers
    std::vector<std::array<uint32_t, 8>> as(100);
    std::vector<std::array<uint32_t, 8>> bs(100);
    std::vector<std::array<uint32_t, 8>> results;
    
    // Initialize with ciphertext indices
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 8; ++j) {
            as[i][j] = i * 16 + j;
            bs[i][j] = i * 16 + j + 8;
        }
    }
    
    auto start = high_resolution_clock::now();
    
    eval.batchAdd8(as, bs, results);
    engine_->sync();
    
    auto end = high_resolution_clock::now();
    double totalMs = duration_cast<milliseconds>(end - start).count();
    
    std::cout << "Batch 8-bit add (100 integers): " << totalMs << " ms" << std::endl;
    std::cout << "Per addition: " << (totalMs / 100) << " ms" << std::endl;
}

//==================================================================================
// Stress Test
//==================================================================================

TEST_F(GPUTFHETest, StressTest) {
    std::cout << "\n=== Stress Test ===" << std::endl;
    std::cout << "Creating 100 users, each with 1000 ciphertexts..." << std::endl;
    
    std::vector<uint64_t> users;
    
    auto startSetup = high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        uint64_t userId = engine_->createUser();
        users.push_back(userId);
        engine_->allocateCiphertexts(userId, 1000);
    }
    
    auto endSetup = high_resolution_clock::now();
    double setupMs = duration_cast<milliseconds>(endSetup - startSetup).count();
    
    std::cout << "Setup time: " << setupMs << " ms" << std::endl;
    std::cout << "GPU memory used: " << (engine_->totalGPUMemoryUsed() / (1024*1024)) << " MB" << std::endl;
    
    // Create massive batch of operations
    std::cout << "Queueing 100,000 operations..." << std::endl;
    
    BatchedGateOp batch;
    batch.gate = GateType::AND;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> idxDist(0, 999);
    std::uniform_int_distribution<size_t> userDist(0, users.size() - 1);
    
    for (int i = 0; i < 100000; ++i) {
        batch.userIds.push_back(users[userDist(gen)]);
        batch.input1Indices.push_back(idxDist(gen));
        batch.input2Indices.push_back(idxDist(gen));
        batch.outputIndices.push_back(idxDist(gen));
    }
    
    auto startOps = high_resolution_clock::now();
    
    engine_->executeBatchGates({batch});
    engine_->sync();
    
    auto endOps = high_resolution_clock::now();
    double opsMs = duration_cast<milliseconds>(endOps - startOps).count();
    
    std::cout << "Execution time: " << opsMs << " ms" << std::endl;
    std::cout << "Throughput: " << (100000 * 1000.0 / opsMs) << " ops/s" << std::endl;
}

//==================================================================================
// Main
//==================================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
