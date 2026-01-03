//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================
#ifndef __BASETESTCASE_H__
#define __BASETESTCASE_H__

#include "config_core.h"
#include "scheme/ckksrns/gen-cryptocontext-ckksrns.h"
#include "scheme/bfvrns/gen-cryptocontext-bfvrns.h"
#include "scheme/bgvrns/gen-cryptocontext-bgvrns.h"
#include "scheme/gen-cryptocontext-params.h"

#include "utils/exception.h"

#include <memory>
#include <string>
#include <vector>

struct BaseTestCase {
private:
    // std::shared_ptr<lux::fhe::Params> params;
    lux::fhe::SCHEME scheme;
    std::vector<std::string> paramOverrides;

public:
    // there are cases when we don't support some features depending on different conditions.
    // skipTest() is to check all those conditions, so we do not get our unit tests failed
    bool skipTest() const {
#if NATIVEINT == 128
        lux::fhe::SCHEME schemeId = lux::fhe::convertToSCHEME(*paramOverrides.begin());
        if (schemeId == lux::fhe::SCHEME::CKKSRNS_SCHEME) {
            lux::fhe::CCParams<lux::fhe::CryptoContextCKKSRNS> parameters(paramOverrides);
            // CKKS does not support FLEXIBLEAUTO or FLEXIBLEAUTOEXT for NATIVEINT == 128
            switch (parameters.GetScalingTechnique()) {
                case lux::fhe::ScalingTechnique::FLEXIBLEAUTO:
                case lux::fhe::ScalingTechnique::FLEXIBLEAUTOEXT:
                    return true;
                default:
                    break;
            }
        }
#endif
        return false;
    }

    // const std::shared_ptr<lux::fhe::Params> getCryptoContextParams() const {
    //    return params;
    // }

    // void setCryptoContextParams(std::shared_ptr<lux::fhe::Params> params0) {
    //    params = params0;
    // }

    const std::vector<std::string>& getCryptoContextParamOverrides() const {
        return paramOverrides;
    }

    /**
     * creates a new cryptocontext parameter object, overrides its data members if necessary and assigns it to params
     *
     * @param vec vector with overrides
     * @return number of all data members of Params or number of vec's elements that can override params
     */
    // size_t populateCryptoContextParams(const std::vector<std::string>::const_iterator& start) {
    //    // get the total number of the parameter override values
    //    size_t numOverrides = lux::fhe::Params::getAllParamsDataMembers().size();

    //    // get the subset of elements with the parameter override values
    //    std::vector<std::string> overrideValues(start, start + numOverrides);

    //    lux::fhe::SCHEME scheme = lux::fhe::convertToSCHEME(*start);
    //    switch (scheme) {
    //    case lux::fhe::CKKSRNS_SCHEME:
    //        setCryptoContextParams(std::make_shared<lux::fhe::CCParams<lux::fhe::CryptoContextCKKSRNS>>(overrideValues));
    //        break;
    //    case lux::fhe::BFVRNS_SCHEME:
    //        setCryptoContextParams(std::make_shared<lux::fhe::CCParams<lux::fhe::CryptoContextBFVRNS>>(overrideValues));
    //        break;
    //    case lux::fhe::BGVRNS_SCHEME:
    //        setCryptoContextParams(std::make_shared<lux::fhe::CCParams<lux::fhe::CryptoContextBGVRNS>>(overrideValues));
    //        break;
    //    default: {
    //        std::string errMsg(std::string("Unknown schemeId ") + std::to_string(scheme));
    //        LUX_FHE_THROW(errMsg);
    //    }
    //    }

    //    return numOverrides;
    //}

    size_t setCryptoContextParamsOverrides(const std::vector<std::string>::const_iterator& start) {
        // get the total number of the parameter override values
        size_t numOverrides = lux::fhe::Params::getAllParamsDataMembers().size();

        scheme = lux::fhe::convertToSCHEME(*start);

        // get the subset of elements with the parameter override values
        try {
            paramOverrides = std::vector<std::string>(start, start + numOverrides);
        }
        catch (...) {
            std::string errMsg("Check the number of parameter overrides in the .csv file. It should be [" +
                               std::to_string(numOverrides) + "]");
            LUX_FHE_THROW(errMsg);
        }
        return numOverrides;
    }
};

#endif  // __BASETESTCASE_H__
