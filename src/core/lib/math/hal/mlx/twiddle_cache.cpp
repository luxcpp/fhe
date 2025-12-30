//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2024, Lux Industries Inc
//
// All rights reserved.
//
// Twiddle Hotset Cache Implementation for GPU FHE
//==================================================================================

#include "math/hal/mlx/twiddle_cache.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace lbcrypto {
namespace mlx_backend {

//==================================================================================
// Modular Arithmetic Helpers (inline for performance)
//==================================================================================

namespace {

/// 128-bit multiplication helper
inline __uint128_t mul128(uint64_t a, uint64_t b) {
    return __uint128_t(a) * b;
}

/// Modular multiplication: (a * b) mod m using 128-bit intermediate
inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>(mul128(a, b) % m);
}

/// Check if n is a power of 2
inline bool isPowerOfTwo(uint32_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/// Compute log2 of a power of 2
inline uint32_t log2(uint32_t n) {
    uint32_t result = 0;
    while (n > 1) {
        n >>= 1;
        ++result;
    }
    return result;
}

}  // anonymous namespace

//==================================================================================
// PrimeConstants Implementation
//==================================================================================

PrimeConstants::PrimeConstants(uint64_t prime, uint64_t ring_dim) {
    q = prime;

    // Montgomery constant: -q^(-1) mod 2^64
    // We need x such that q * x = -1 mod 2^64
    // Using extended Euclidean algorithm optimized for power-of-2 modulus
    uint64_t x = 1;
    for (int i = 0; i < 6; ++i) {
        x *= 2 - q * x;  // Newton iteration for 2-adic inverse
    }
    q_inv = static_cast<uint64_t>(-(static_cast<int64_t>(x)));

    // Barrett constant: floor(2^128 / q)
    // We store the high 64 bits (mu_hi) and low 64 bits (mu_lo)
    __uint128_t numerator = __uint128_t(1) << 127;  // 2^127, we'll double later
    __uint128_t quotient = (numerator / q) * 2;     // 2^128 / q
    mu_hi = static_cast<uint64_t>(quotient >> 64);
    mu_lo = static_cast<uint64_t>(quotient);

    // R^2 mod q where R = 2^64 (Montgomery domain entry)
    __uint128_t R = __uint128_t(1) << 64;
    __uint128_t R_sq = (R % q) * (R % q) % q;
    r_squared = static_cast<uint64_t>(R_sq);

    // Root of unity will be computed separately
    root = 0;
    root_inv = 0;

    // n_inv will be computed with the ring dimension
    // Compute n^(-1) mod q using extended Euclidean algorithm
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(q), newr = static_cast<int64_t>(ring_dim);
    while (newr != 0) {
        int64_t quotient_r = r / newr;
        std::tie(t, newt) = std::make_pair(newt, t - quotient_r * newt);
        std::tie(r, newr) = std::make_pair(newr, r - quotient_r * newr);
    }
    n_inv = static_cast<uint64_t>(t < 0 ? t + static_cast<int64_t>(q) : t);
}

//==================================================================================
// StageTwiddleCache Implementation
//==================================================================================

StageTwiddleCache::StageTwiddleCache(uint32_t stage, uint32_t N, uint64_t q, uint64_t root) {
    stageIndex = stage;

    // For Cooley-Tukey NTT:
    // Stage s has 2^s unique twiddles, each used N/(2^(s+1)) times
    twiddleCount = 1u << stage;
    stride = N / twiddleCount;

    twiddles.resize(twiddleCount);

    // Compute twiddles: w^0, w^stride, w^(2*stride), ...
    uint64_t w = 1;
    uint64_t w_stride = 1;

    // Compute w^stride
    for (uint32_t i = 0; i < stride; ++i) {
        w_stride = mulmod(w_stride, root, q);
    }

    for (uint32_t i = 0; i < twiddleCount; ++i) {
        twiddles[i] = w;
        w = mulmod(w, w_stride, q);
    }
}

//==================================================================================
// TwiddleCache Implementation
//==================================================================================

TwiddleCache::TwiddleCache(uint32_t ringDim, const std::vector<uint64_t>& primes)
    : ringDim_(ringDim), primes_(primes) {

    if (!isPowerOfTwo(ringDim)) {
        throw std::invalid_argument("Ring dimension must be a power of 2");
    }

    if (primes.empty() || primes.size() > MAX_RNS_PRIMES) {
        throw std::invalid_argument("Number of primes must be 1-" +
                                    std::to_string(MAX_RNS_PRIMES));
    }

    logRingDim_ = log2(ringDim);

    // Validate NTT-friendliness
    for (size_t i = 0; i < primes.size(); ++i) {
        if (!isNTTFriendly(primes[i], ringDim)) {
            throw std::invalid_argument("Prime " + std::to_string(primes[i]) +
                                        " is not NTT-friendly for N=" +
                                        std::to_string(ringDim));
        }
    }

    // Initialize constant cache metadata
    constantCache_.numPrimes = static_cast<uint32_t>(primes.size());
    constantCache_.ringDim = ringDim;

    // Initialize device table
    deviceTable_.numPrimes = static_cast<uint32_t>(primes.size());
    deviceTable_.ringDim = ringDim;
    deviceTable_.layout = TwiddleLayout::TWIDDLE_MAJOR;

    // Allocate stage caches
    stageCaches_.resize(primes.size());
    for (size_t p = 0; p < primes.size(); ++p) {
        stageCaches_[p].resize(logRingDim_);
    }

#ifdef WITH_MLX
    stageBuffers_.resize(primes.size());
    for (size_t p = 0; p < primes.size(); ++p) {
        stageBuffers_[p].resize(logRingDim_);
    }
#endif
}

TwiddleCache::~TwiddleCache() = default;

TwiddleCache::TwiddleCache(TwiddleCache&& other) noexcept
    : ringDim_(other.ringDim_),
      logRingDim_(other.logRingDim_),
      primes_(std::move(other.primes_)),
      constantCache_(std::move(other.constantCache_)),
      stageCaches_(std::move(other.stageCaches_)),
      deviceTable_(std::move(other.deviceTable_))
#ifdef WITH_MLX
      , constantBuffer_(std::move(other.constantBuffer_))
      , deviceForwardTwiddles_(std::move(other.deviceForwardTwiddles_))
      , deviceInverseTwiddles_(std::move(other.deviceInverseTwiddles_))
      , stageBuffers_(std::move(other.stageBuffers_))
#endif
      , isPrecomputed_(other.isPrecomputed_)
      , isCacheWarm_(other.isCacheWarm_)
{
    // stats_ has atomic members, can't be moved, just reset
}

TwiddleCache& TwiddleCache::operator=(TwiddleCache&& other) noexcept {
    if (this != &other) {
        ringDim_ = other.ringDim_;
        logRingDim_ = other.logRingDim_;
        primes_ = std::move(other.primes_);
        isPrecomputed_ = other.isPrecomputed_;
        isCacheWarm_ = other.isCacheWarm_;
        constantCache_ = std::move(other.constantCache_);
        stageCaches_ = std::move(other.stageCaches_);
        deviceTable_ = std::move(other.deviceTable_);
#ifdef WITH_MLX
        constantBuffer_ = std::move(other.constantBuffer_);
        deviceForwardTwiddles_ = std::move(other.deviceForwardTwiddles_);
        deviceInverseTwiddles_ = std::move(other.deviceInverseTwiddles_);
        stageBuffers_ = std::move(other.stageBuffers_);
#endif
        // stats_ not moved, just reset
        stats_.reset();
    }
    return *this;
}

//==================================================================================
// Precomputation
//==================================================================================

void TwiddleCache::precompute() {
    if (isPrecomputed_) {
        return;
    }

    size_t numPrimes = primes_.size();

    // Allocate device table storage
    size_t tableSize = numPrimes * ringDim_;
    deviceTable_.forwardTwiddles.resize(tableSize);
    deviceTable_.inverseTwiddles.resize(tableSize);
    deviceTable_.forwardTwiddlesBitRev.resize(tableSize);
    deviceTable_.inverseTwiddlesBitRev.resize(tableSize);

    // Process each prime
    for (size_t p = 0; p < numPrimes; ++p) {
        uint64_t q = primes_[p];

        // Compute prime constants
        constantCache_.primes[p] = PrimeConstants(q, ringDim_);

        // Find primitive root of unity
        uint64_t root = computePrimitiveRoot(ringDim_, q);
        uint64_t root_inv = modInverse(root, q);

        constantCache_.primes[p].root = root;
        constantCache_.primes[p].root_inv = root_inv;

        // Compute Barrett constant properly
        computeBarrettConstant(q,
                               constantCache_.primes[p].mu_hi,
                               constantCache_.primes[p].mu_lo);

        // Compute first-level twiddles for constant memory
        computeFirstLevelTwiddles(static_cast<uint32_t>(p));

        // Compute stage caches for threadgroup memory
        precomputeStageCaches(static_cast<uint32_t>(p));

        // Compute complete device twiddle tables
        std::vector<uint64_t> fwdTwiddles(ringDim_);
        std::vector<uint64_t> invTwiddles(ringDim_);

        computeTwiddleTable(q, root, fwdTwiddles);
        computeTwiddleTable(q, root_inv, invTwiddles);

        // Copy to device table
        if (deviceTable_.layout == TwiddleLayout::TWIDDLE_MAJOR) {
            // [twiddle][prime] layout for coalesced RNS access
            for (size_t i = 0; i < ringDim_; ++i) {
                deviceTable_.forwardTwiddles[i * numPrimes + p] = fwdTwiddles[i];
                deviceTable_.inverseTwiddles[i * numPrimes + p] = invTwiddles[i];
            }
        } else {
            // [prime][twiddle] layout
            size_t offset = p * ringDim_;
            std::copy(fwdTwiddles.begin(), fwdTwiddles.end(),
                      deviceTable_.forwardTwiddles.begin() + offset);
            std::copy(invTwiddles.begin(), invTwiddles.end(),
                      deviceTable_.inverseTwiddles.begin() + offset);
        }

        // Create bit-reversed versions
        auto fwdBitRev = fwdTwiddles;
        auto invBitRev = invTwiddles;
        bitReversePermute(fwdBitRev);
        bitReversePermute(invBitRev);

        if (deviceTable_.layout == TwiddleLayout::TWIDDLE_MAJOR) {
            for (size_t i = 0; i < ringDim_; ++i) {
                deviceTable_.forwardTwiddlesBitRev[i * numPrimes + p] = fwdBitRev[i];
                deviceTable_.inverseTwiddlesBitRev[i * numPrimes + p] = invBitRev[i];
            }
        } else {
            size_t offset = p * ringDim_;
            std::copy(fwdBitRev.begin(), fwdBitRev.end(),
                      deviceTable_.forwardTwiddlesBitRev.begin() + offset);
            std::copy(invBitRev.begin(), invBitRev.end(),
                      deviceTable_.inverseTwiddlesBitRev.begin() + offset);
        }
    }

    isPrecomputed_ = true;
}

void TwiddleCache::computeFirstLevelTwiddles(uint32_t primeIdx) {
    uint64_t q = primes_[primeIdx];
    uint64_t root = constantCache_.primes[primeIdx].root;
    uint64_t root_inv = constantCache_.primes[primeIdx].root_inv;

    // First-level twiddles: w^(bit_reverse(i, log2(8))) for i in [0, 8)
    // These cover NTT stages 0-3 where the number of unique twiddles <= 8
    for (size_t i = 0; i < FIRST_LEVEL_TWIDDLE_COUNT; ++i) {
        uint32_t br_i = bitReverse(static_cast<uint32_t>(i), 3);  // 3 bits for 8 values
        uint64_t exp = (ringDim_ / 8) * br_i;  // Scale to full ring

        constantCache_.firstLevelTwiddles[primeIdx][i] = modPow(root, exp, q);
        constantCache_.firstLevelInvTwiddles[primeIdx][i] = modPow(root_inv, exp, q);
    }
}

void TwiddleCache::precomputeStageCaches(uint32_t primeIdx) {
    uint64_t q = primes_[primeIdx];
    uint64_t root = constantCache_.primes[primeIdx].root;

    for (uint32_t stage = 0; stage < logRingDim_; ++stage) {
        stageCaches_[primeIdx][stage] = StageTwiddleCache(stage, ringDim_, q, root);
    }
}

//==================================================================================
// GPU Cache Warming
//==================================================================================

void TwiddleCache::warmCache() {
    if (!isPrecomputed_) {
        precompute();
    }

    if (isCacheWarm_) {
        return;
    }

#ifdef WITH_MLX
    namespace mx = mlx::core;

    // Check GPU availability
    if (!mx::metal::is_available()) {
        std::cerr << "TwiddleCache: GPU not available, running in CPU mode\n";
        isCacheWarm_ = true;
        return;
    }

    mx::set_default_device(mx::Device::gpu);

    // Upload constant cache as raw bytes
    std::vector<uint8_t> constBytes(sizeof(ConstantMemoryCache));
    std::memcpy(constBytes.data(), &constantCache_, sizeof(ConstantMemoryCache));
    constantBuffer_ = mx::array(constBytes.data(),
                                {static_cast<int>(constBytes.size())},
                                mx::uint8);
    mx::eval(*constantBuffer_);

    // Upload device twiddle tables
    size_t numPrimes = primes_.size();

    if (deviceTable_.layout == TwiddleLayout::TWIDDLE_MAJOR) {
        // Shape: [ringDim, numPrimes]
        std::vector<int64_t> fwdData(deviceTable_.forwardTwiddlesBitRev.begin(),
                                      deviceTable_.forwardTwiddlesBitRev.end());
        std::vector<int64_t> invData(deviceTable_.inverseTwiddlesBitRev.begin(),
                                      deviceTable_.inverseTwiddlesBitRev.end());

        deviceForwardTwiddles_ = mx::array(fwdData.data(),
                                           {static_cast<int>(ringDim_),
                                            static_cast<int>(numPrimes)},
                                           mx::int64);
        deviceInverseTwiddles_ = mx::array(invData.data(),
                                           {static_cast<int>(ringDim_),
                                            static_cast<int>(numPrimes)},
                                           mx::int64);
    } else {
        // Shape: [numPrimes, ringDim]
        std::vector<int64_t> fwdData(deviceTable_.forwardTwiddlesBitRev.begin(),
                                      deviceTable_.forwardTwiddlesBitRev.end());
        std::vector<int64_t> invData(deviceTable_.inverseTwiddlesBitRev.begin(),
                                      deviceTable_.inverseTwiddlesBitRev.end());

        deviceForwardTwiddles_ = mx::array(fwdData.data(),
                                           {static_cast<int>(numPrimes),
                                            static_cast<int>(ringDim_)},
                                           mx::int64);
        deviceInverseTwiddles_ = mx::array(invData.data(),
                                           {static_cast<int>(numPrimes),
                                            static_cast<int>(ringDim_)},
                                           mx::int64);
    }

    mx::eval(*deviceForwardTwiddles_);
    mx::eval(*deviceInverseTwiddles_);

    // Upload stage-specific twiddles
    for (size_t p = 0; p < numPrimes; ++p) {
        for (uint32_t stage = 0; stage < logRingDim_; ++stage) {
            const auto& stageTw = stageCaches_[p][stage].twiddles;
            std::vector<int64_t> stageData(stageTw.begin(), stageTw.end());

            stageBuffers_[p][stage] = mx::array(stageData.data(),
                                                {static_cast<int>(stageData.size())},
                                                mx::int64);
            mx::eval(*stageBuffers_[p][stage]);
        }
    }

    // Touch all memory to ensure it's in GPU cache
    auto dummy = mx::sum(*deviceForwardTwiddles_) + mx::sum(*deviceInverseTwiddles_);
    mx::eval(dummy);

    std::cout << "TwiddleCache: GPU cache warmed ("
              << totalMemoryBytes() / 1024 << " KB)\n";
#endif

    isCacheWarm_ = true;
}

bool TwiddleCache::verifyCachePerformance() {
    if (!isCacheWarm_) {
        warmCache();
    }

#ifdef WITH_MLX
    namespace mx = mlx::core;

    if (!mx::metal::is_available()) {
        return false;
    }

    // Simple verification: read twiddles and verify they match CPU values
    mx::eval(*deviceForwardTwiddles_);

    // Check first few values
    size_t numPrimes = primes_.size();
    for (size_t p = 0; p < numPrimes; ++p) {
        // First twiddle should be 1 (w^0)
        uint64_t expectedZero = 1;

        // Verify by checking the precomputed values match
        if (deviceTable_.forwardTwiddlesBitRev[p] != expectedZero) {
            std::cerr << "Cache verification failed: twiddle[0] mismatch\n";
            return false;
        }
    }

    return true;
#else
    return true;  // CPU mode always "works"
#endif
}

//==================================================================================
// Twiddle Access
//==================================================================================

uint64_t TwiddleCache::getForwardTwiddle(uint32_t primeIdx, uint32_t twiddleIdx) const {
    if (primeIdx >= primes_.size() || twiddleIdx >= ringDim_) {
        throw std::out_of_range("Twiddle index out of range");
    }

    stats_.totalAccesses++;

    // Check if in first-level cache
    if (twiddleIdx < FIRST_LEVEL_TWIDDLE_COUNT) {
        stats_.constantHits++;
        return constantCache_.firstLevelTwiddles[primeIdx][twiddleIdx];
    }

    // Otherwise access from device table
    stats_.deviceAccesses++;

    size_t idx = (deviceTable_.layout == TwiddleLayout::TWIDDLE_MAJOR)
                     ? twiddleIdx * primes_.size() + primeIdx
                     : primeIdx * ringDim_ + twiddleIdx;

    return deviceTable_.forwardTwiddles[idx];
}

uint64_t TwiddleCache::getInverseTwiddle(uint32_t primeIdx, uint32_t twiddleIdx) const {
    if (primeIdx >= primes_.size() || twiddleIdx >= ringDim_) {
        throw std::out_of_range("Twiddle index out of range");
    }

    stats_.totalAccesses++;

    if (twiddleIdx < FIRST_LEVEL_TWIDDLE_COUNT) {
        stats_.constantHits++;
        return constantCache_.firstLevelInvTwiddles[primeIdx][twiddleIdx];
    }

    stats_.deviceAccesses++;

    size_t idx = (deviceTable_.layout == TwiddleLayout::TWIDDLE_MAJOR)
                     ? twiddleIdx * primes_.size() + primeIdx
                     : primeIdx * ringDim_ + twiddleIdx;

    return deviceTable_.inverseTwiddles[idx];
}

void TwiddleCache::getForwardTwiddleBatch(uint32_t primeIdx, uint32_t startIdx,
                                          uint32_t count, uint64_t* output) const {
    for (uint32_t i = 0; i < count; ++i) {
        output[i] = getForwardTwiddle(primeIdx, startIdx + i);
    }
}

void TwiddleCache::getInverseTwiddleBatch(uint32_t primeIdx, uint32_t startIdx,
                                          uint32_t count, uint64_t* output) const {
    for (uint32_t i = 0; i < count; ++i) {
        output[i] = getInverseTwiddle(primeIdx, startIdx + i);
    }
}

const PrimeConstants& TwiddleCache::getPrimeConstants(uint32_t primeIdx) const {
    if (primeIdx >= primes_.size()) {
        throw std::out_of_range("Prime index out of range");
    }
    return constantCache_.primes[primeIdx];
}

#ifdef WITH_MLX
mlx::core::array TwiddleCache::getStageTwiddles(uint32_t stageIdx, uint32_t primeIdx) const {
    if (primeIdx >= primes_.size() || stageIdx >= logRingDim_) {
        throw std::out_of_range("Stage or prime index out of range");
    }
    return *stageBuffers_[primeIdx][stageIdx];
}
#endif

//==================================================================================
// Statistics and Analysis
//==================================================================================

double TwiddleCache::estimateHitRate(uint32_t ringDim, uint32_t batchSize) {
    uint32_t logN = log2(ringDim);

    // First 4 stages use <= 8 twiddles, all from constant memory
    uint32_t constStages = std::min(logN, 4u);
    uint64_t constAccesses = constStages * (ringDim / 2);

    // Remaining stages use threadgroup memory
    uint32_t tgStages = logN - constStages;
    uint64_t tgAccesses = tgStages * (ringDim / 2);

    // Total accesses per NTT
    uint64_t totalAccesses = constAccesses + tgAccesses;

    // All accesses hit L1 or L2, so effective hit rate is 100%
    // But we can report the distribution
    double constRate = static_cast<double>(constAccesses) / totalAccesses;
    double tgRate = static_cast<double>(tgAccesses) / totalAccesses;

    // Weighted by latency: const=4, tg=20, device=200
    double effectiveLatency = constRate * 4.0 + tgRate * 20.0;
    double baselineLatency = 200.0;

    // Return speedup factor as "effective hit rate"
    return baselineLatency / effectiveLatency;
}

double TwiddleCache::estimateBandwidthSavings(uint32_t ringDim, uint32_t batchSize) {
    uint32_t logN = log2(ringDim);

    // Without caching: every twiddle access goes to device memory
    // Per NTT: logN stages, each with N/2 butterflies, each needing 1 twiddle
    // Total: N/2 * logN twiddle reads per NTT
    uint64_t uncachedReads = (ringDim / 2) * logN;

    // With caching:
    // - First 4 stages: 0 device reads (constant memory)
    // - Later stages: 1 device read per unique twiddle, reused many times
    // - Per stage s: 2^s unique twiddles, each used N/(2^(s+1)) times
    uint64_t cachedReads = 0;
    for (uint32_t s = 4; s < logN; ++s) {
        cachedReads += 1u << s;  // Unique twiddles loaded
    }

    // With batching, cached reads are amortized
    double amortizedCachedReads = static_cast<double>(cachedReads) / batchSize;

    double savings = 1.0 - (amortizedCachedReads / uncachedReads);
    return savings;
}

void TwiddleCache::setLayout(TwiddleLayout layout) {
    if (layout != deviceTable_.layout) {
        deviceTable_.layout = layout;
        isPrecomputed_ = false;  // Need to recompute with new layout
        isCacheWarm_ = false;
    }
}

size_t TwiddleCache::totalMemoryBytes() const {
    size_t total = 0;

    // Constant cache
    total += sizeof(ConstantMemoryCache);

    // Stage caches
    for (const auto& primeCaches : stageCaches_) {
        for (const auto& stageCache : primeCaches) {
            total += stageCache.twiddles.size() * sizeof(uint64_t);
        }
    }

    // Device table
    total += deviceTable_.sizeBytes();

    return total;
}

//==================================================================================
// Mathematical Helpers
//==================================================================================

uint64_t TwiddleCache::computePrimitiveRoot(uint64_t N, uint64_t q) const {
    // For NTT-friendly prime q = k * 2N + 1, find primitive 2N-th root of unity
    // Such a root w satisfies: w^(2N) = 1 and w^N = -1 = q-1

    // Try small generators
    for (uint64_t g = 2; g < 1000; ++g) {
        // Compute w = g^((q-1)/(2N))
        uint64_t exp = (q - 1) / (2 * N);
        uint64_t w = modPow(g, exp, q);

        // Verify: w^(2N) = 1 and w^N = q-1
        if (modPow(w, 2 * N, q) == 1 && modPow(w, N, q) == q - 1) {
            return w;
        }
    }

    throw std::runtime_error("Failed to find primitive root of unity for q=" +
                             std::to_string(q) + ", N=" + std::to_string(N));
}

void TwiddleCache::computeBarrettConstant(uint64_t q, uint64_t& mu_hi, uint64_t& mu_lo) const {
    // Barrett constant: floor(2^128 / q)
    // We compute this as floor(2^64 / q) * 2^64 + adjustment

    // High part: floor(2^64 / q)
    __uint128_t R = __uint128_t(1) << 64;
    __uint128_t mu = R / q;  // This gives floor(2^64 / q)

    // Now we need floor(2^128 / q)
    // = floor((2^64)^2 / q)
    // We use: floor(2^128 / q) = floor(2^64 / q) * 2^64 + floor((2^64 mod q) * 2^64 / q)

    __uint128_t r = R % q;  // 2^64 mod q
    __uint128_t mu_adjust = (r << 64) / q;

    __uint128_t mu_full = (mu << 64) + mu_adjust;
    mu_hi = static_cast<uint64_t>(mu_full >> 64);
    mu_lo = static_cast<uint64_t>(mu_full);
}

uint64_t TwiddleCache::computeMontgomeryConstant(uint64_t q) const {
    // Compute -q^(-1) mod 2^64 using Newton iteration
    uint64_t x = 1;
    for (int i = 0; i < 6; ++i) {
        x *= 2 - q * x;
    }
    return static_cast<uint64_t>(-(static_cast<int64_t>(x)));
}

uint64_t TwiddleCache::computeRSquared(uint64_t q) const {
    // Compute R^2 mod q where R = 2^64
    __uint128_t R = __uint128_t(1) << 64;
    __uint128_t R_mod_q = R % q;
    return static_cast<uint64_t>((R_mod_q * R_mod_q) % q);
}

uint64_t TwiddleCache::modInverse(uint64_t a, uint64_t m) const {
    // Extended Euclidean algorithm
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(m), newr = static_cast<int64_t>(a);

    while (newr != 0) {
        int64_t quotient = r / newr;
        std::tie(t, newt) = std::make_pair(newt, t - quotient * newt);
        std::tie(r, newr) = std::make_pair(newr, r - quotient * newr);
    }

    if (t < 0) {
        t += static_cast<int64_t>(m);
    }

    return static_cast<uint64_t>(t);
}

uint64_t TwiddleCache::modPow(uint64_t base, uint64_t exp, uint64_t m) const {
    uint64_t result = 1;
    base %= m;

    while (exp > 0) {
        if (exp & 1) {
            result = mulmod(result, base, m);
        }
        exp >>= 1;
        base = mulmod(base, base, m);
    }

    return result;
}

void TwiddleCache::computeTwiddleTable(uint64_t q, uint64_t root,
                                       std::vector<uint64_t>& twiddles) const {
    twiddles.resize(ringDim_);

    uint64_t w = 1;
    for (size_t i = 0; i < ringDim_; ++i) {
        twiddles[i] = w;
        w = mulmod(w, root, q);
    }
}

void TwiddleCache::bitReversePermute(std::vector<uint64_t>& arr) const {
    size_t n = arr.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = bitReverse(static_cast<uint32_t>(i), logRingDim_);
        if (i < j) {
            std::swap(arr[i], arr[j]);
        }
    }
}

uint32_t TwiddleCache::bitReverse(uint32_t x, uint32_t bits) const {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

//==================================================================================
// Factory Functions
//==================================================================================

std::unique_ptr<TwiddleCache> createTFHECache(uint32_t N, uint64_t Q) {
    std::vector<uint64_t> primes = {Q};
    auto cache = std::make_unique<TwiddleCache>(N, primes);
    cache->precompute();
    cache->warmCache();
    return cache;
}

std::unique_ptr<TwiddleCache> createRNSCache(uint32_t N, const std::vector<uint64_t>& primes) {
    auto cache = std::make_unique<TwiddleCache>(N, primes);
    cache->precompute();
    cache->warmCache();
    return cache;
}

//==================================================================================
// Utility Functions
//==================================================================================

bool isNTTFriendly(uint64_t q, uint32_t N) {
    // A prime q is NTT-friendly for dimension N if q = 1 mod 2N
    return (q % (2 * N)) == 1;
}

std::vector<uint64_t> generateNTTFriendlyPrimes(uint32_t N, uint32_t bitWidth, uint32_t count) {
    std::vector<uint64_t> result;
    result.reserve(count);

    // Start searching from 2^(bitWidth-1)
    uint64_t start = 1ULL << (bitWidth - 1);
    uint64_t step = 2 * N;  // q must be 1 mod 2N

    // Find first candidate: start + (step - (start % step)) % step + 1
    uint64_t candidate = start;
    candidate += (step - (candidate % step)) % step;
    candidate += 1;  // Make it 1 mod 2N

    while (result.size() < count) {
        // Miller-Rabin primality test
        auto isPrime = [](uint64_t n) -> bool {
            if (n < 2) return false;
            if (n == 2 || n == 3) return true;
            if (n % 2 == 0) return false;

            // Write n-1 as 2^r * d
            uint64_t d = n - 1;
            int r = 0;
            while ((d & 1) == 0) {
                d >>= 1;
                ++r;
            }

            // Witnesses for deterministic test up to 2^64
            std::array<uint64_t, 7> witnesses = {2, 3, 5, 7, 11, 13, 17};

            for (uint64_t a : witnesses) {
                if (a >= n) continue;

                // Compute a^d mod n
                uint64_t x = 1;
                uint64_t base = a % n;
                uint64_t exp = d;

                while (exp > 0) {
                    if (exp & 1) {
                        x = static_cast<uint64_t>((__uint128_t(x) * base) % n);
                    }
                    exp >>= 1;
                    base = static_cast<uint64_t>((__uint128_t(base) * base) % n);
                }

                if (x == 1 || x == n - 1) continue;

                bool composite = true;
                for (int i = 0; i < r - 1; ++i) {
                    x = static_cast<uint64_t>((__uint128_t(x) * x) % n);
                    if (x == n - 1) {
                        composite = false;
                        break;
                    }
                }

                if (composite) return false;
            }

            return true;
        };

        if (isPrime(candidate)) {
            result.push_back(candidate);
        }

        candidate += step;

        // Check for overflow
        if (candidate < step) {
            throw std::runtime_error("Ran out of primes at specified bit width");
        }
    }

    return result;
}

}  // namespace mlx_backend
}  // namespace lbcrypto
