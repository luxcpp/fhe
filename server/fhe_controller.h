/**
 *  FHE Controller - HTTP/gRPC endpoints for FHE operations
 */

#pragma once

#include <http/HttpController.h>
#include <http/HttpAppFramework.h>
#include <binfhecontext.h>
#include <mutex>
#include <unordered_map>
#include <atomic>

namespace fhe
{

using namespace http;
using namespace lbcrypto;

/**
 * @brief Manages FHE contexts and keys
 */
class FHEManager
{
  public:
    static FHEManager& instance()
    {
        static FHEManager inst;
        return inst;
    }

    // Create a new context and return its ID
    std::string createContext(BINFHE_PARAMSET paramSet = STD128_LMKCDEY,
                              BINFHE_METHOD method = LMKCDEY)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto ctx = std::make_shared<BinFHEContext>();
        ctx->GenerateBinFHEContext(paramSet, method);
        
        std::string id = "ctx_" + std::to_string(nextId_++);
        contexts_[id] = ctx;
        
        LOG_INFO << "Created FHE context: " << id;
        return id;
    }

    // Generate keys for a context
    bool generateKeys(const std::string& contextId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = contexts_.find(contextId);
        if (it == contexts_.end()) return false;
        
        auto& ctx = it->second;
        auto sk = ctx->KeyGen();
        ctx->BTKeyGen(sk);
        
        secretKeys_[contextId] = sk;
        LOG_INFO << "Generated keys for context: " << contextId;
        return true;
    }

    // Encrypt a boolean value
    std::string encrypt(const std::string& contextId, bool value, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto ctxIt = contexts_.find(contextId);
        auto skIt = secretKeys_.find(contextId);
        
        if (ctxIt == contexts_.end()) {
            error = "Context not found";
            return "";
        }
        if (skIt == secretKeys_.end()) {
            error = "Keys not generated";
            return "";
        }
        
        auto ct = ctxIt->second->Encrypt(skIt->second, value ? 1 : 0);
        
        std::string ctId = "ct_" + std::to_string(nextId_++);
        ciphertexts_[ctId] = ct;
        ciphertextContext_[ctId] = contextId;
        
        return ctId;
    }

    // Decrypt a ciphertext
    bool decrypt(const std::string& ciphertextId, bool& result, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto ctIt = ciphertexts_.find(ciphertextId);
        if (ctIt == ciphertexts_.end()) {
            error = "Ciphertext not found";
            return false;
        }
        
        auto& contextId = ciphertextContext_[ciphertextId];
        auto ctxIt = contexts_.find(contextId);
        auto skIt = secretKeys_.find(contextId);
        
        if (ctxIt == contexts_.end() || skIt == secretKeys_.end()) {
            error = "Context or keys not found";
            return false;
        }
        
        LWEPlaintext pt;
        ctxIt->second->Decrypt(skIt->second, ctIt->second, &pt);
        result = (pt != 0);
        
        return true;
    }

    // Evaluate AND gate
    std::string evalAnd(const std::string& ct1Id, const std::string& ct2Id, std::string& error)
    {
        return evalBinaryGate(ct1Id, ct2Id, AND, error);
    }

    // Evaluate OR gate
    std::string evalOr(const std::string& ct1Id, const std::string& ct2Id, std::string& error)
    {
        return evalBinaryGate(ct1Id, ct2Id, OR, error);
    }

    // Evaluate XOR gate
    std::string evalXor(const std::string& ct1Id, const std::string& ct2Id, std::string& error)
    {
        return evalBinaryGate(ct1Id, ct2Id, XOR_FAST, error);
    }

    // Evaluate NAND gate
    std::string evalNand(const std::string& ct1Id, const std::string& ct2Id, std::string& error)
    {
        return evalBinaryGate(ct1Id, ct2Id, NAND, error);
    }

    // Evaluate NOR gate
    std::string evalNor(const std::string& ct1Id, const std::string& ct2Id, std::string& error)
    {
        return evalBinaryGate(ct1Id, ct2Id, NOR, error);
    }

    // Evaluate XNOR gate
    std::string evalXnor(const std::string& ct1Id, const std::string& ct2Id, std::string& error)
    {
        return evalBinaryGate(ct1Id, ct2Id, XNOR_FAST, error);
    }

    // Evaluate NOT gate
    std::string evalNot(const std::string& ctId, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto ctIt = ciphertexts_.find(ctId);
        if (ctIt == ciphertexts_.end()) {
            error = "Ciphertext not found";
            return "";
        }
        
        auto& contextId = ciphertextContext_[ctId];
        auto ctxIt = contexts_.find(contextId);
        
        if (ctxIt == contexts_.end()) {
            error = "Context not found";
            return "";
        }
        
        auto result = ctxIt->second->EvalNOT(ctIt->second);
        
        std::string resultId = "ct_" + std::to_string(nextId_++);
        ciphertexts_[resultId] = result;
        ciphertextContext_[resultId] = contextId;
        
        return resultId;
    }

    // Get stats
    Json::Value getStats()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Json::Value stats;
        stats["contexts"] = static_cast<Json::UInt64>(contexts_.size());
        stats["ciphertexts"] = static_cast<Json::UInt64>(ciphertexts_.size());
        return stats;
    }

  private:
    FHEManager() = default;

    std::string evalBinaryGate(const std::string& ct1Id, 
                               const std::string& ct2Id,
                               BINGATE gate,
                               std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto ct1It = ciphertexts_.find(ct1Id);
        auto ct2It = ciphertexts_.find(ct2Id);
        
        if (ct1It == ciphertexts_.end() || ct2It == ciphertexts_.end()) {
            error = "Ciphertext not found";
            return "";
        }
        
        auto& contextId = ciphertextContext_[ct1Id];
        auto ctxIt = contexts_.find(contextId);
        
        if (ctxIt == contexts_.end()) {
            error = "Context not found";
            return "";
        }
        
        auto result = ctxIt->second->EvalBinGate(gate, ct1It->second, ct2It->second);
        
        std::string resultId = "ct_" + std::to_string(nextId_++);
        ciphertexts_[resultId] = result;
        ciphertextContext_[resultId] = contextId;
        
        return resultId;
    }

    std::mutex mutex_;
    std::atomic<uint64_t> nextId_{1};
    std::unordered_map<std::string, std::shared_ptr<BinFHEContext>> contexts_;
    std::unordered_map<std::string, LWEPrivateKey> secretKeys_;
    std::unordered_map<std::string, LWECiphertext> ciphertexts_;
    std::unordered_map<std::string, std::string> ciphertextContext_;
};

/**
 * @brief HTTP Controller for FHE operations
 */
class FHEController : public HttpController<FHEController>
{
  public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(FHEController::health, "/health", Get);
    ADD_METHOD_TO(FHEController::stats, "/v1/stats", Get);
    ADD_METHOD_TO(FHEController::createContext, "/v1/context/create", Post);
    ADD_METHOD_TO(FHEController::generateKeys, "/v1/keys/generate", Post);
    ADD_METHOD_TO(FHEController::encrypt, "/v1/encrypt", Post);
    ADD_METHOD_TO(FHEController::decrypt, "/v1/decrypt", Post);
    ADD_METHOD_TO(FHEController::evalAnd, "/v1/eval/and", Post);
    ADD_METHOD_TO(FHEController::evalOr, "/v1/eval/or", Post);
    ADD_METHOD_TO(FHEController::evalXor, "/v1/eval/xor", Post);
    ADD_METHOD_TO(FHEController::evalNot, "/v1/eval/not", Post);
    ADD_METHOD_TO(FHEController::evalNand, "/v1/eval/nand", Post);
    ADD_METHOD_TO(FHEController::evalNor, "/v1/eval/nor", Post);
    ADD_METHOD_TO(FHEController::evalXnor, "/v1/eval/xnor", Post);
    METHOD_LIST_END

    void health(const HttpRequestPtr& req,
                std::function<void(const HttpResponsePtr&)>&& callback)
    {
        Json::Value resp;
        resp["status"] = "healthy";
        resp["service"] = "lux-fhe";
        resp["version"] = "1.0.0";
        callback(HttpResponse::newHttpJsonResponse(resp));
    }

    void stats(const HttpRequestPtr& req,
               std::function<void(const HttpResponsePtr&)>&& callback)
    {
        auto stats = FHEManager::instance().getStats();
        callback(HttpResponse::newHttpJsonResponse(stats));
    }

    void createContext(const HttpRequestPtr& req,
                       std::function<void(const HttpResponsePtr&)>&& callback)
    {
        auto json = req->getJsonObject();
        
        BINFHE_PARAMSET paramSet = STD128_LMKCDEY;
        BINFHE_METHOD method = LMKCDEY;
        
        std::string securityName = "STD128_LMKCDEY";
        std::string methodName = "LMKCDEY";
        
        if (json) {
            if (json->isMember("security")) {
                std::string sec = (*json)["security"].asString();
                securityName = sec;
                // 128-bit classical
                if (sec == "STD128") paramSet = STD128;
                else if (sec == "STD128_LMKCDEY") paramSet = STD128_LMKCDEY;
                else if (sec == "STD128_3_LMKCDEY") paramSet = STD128_3_LMKCDEY;
                else if (sec == "STD128_4_LMKCDEY") paramSet = STD128_4_LMKCDEY;
                // 128-bit quantum
                else if (sec == "STD128Q") paramSet = STD128Q;
                else if (sec == "STD128Q_LMKCDEY") paramSet = STD128Q_LMKCDEY;
                else if (sec == "STD128Q_3_LMKCDEY") paramSet = STD128Q_3_LMKCDEY;
                else if (sec == "STD128Q_4_LMKCDEY") paramSet = STD128Q_4_LMKCDEY;
                // 192-bit classical
                else if (sec == "STD192") paramSet = STD192;
                else if (sec == "STD192_LMKCDEY") paramSet = STD192_LMKCDEY;
                else if (sec == "STD192_3_LMKCDEY") paramSet = STD192_3_LMKCDEY;
                else if (sec == "STD192_4_LMKCDEY") paramSet = STD192_4_LMKCDEY;
                // 192-bit quantum
                else if (sec == "STD192Q") paramSet = STD192Q;
                else if (sec == "STD192Q_LMKCDEY") paramSet = STD192Q_LMKCDEY;
                else if (sec == "STD192Q_3_LMKCDEY") paramSet = STD192Q_3_LMKCDEY;
                else if (sec == "STD192Q_4_LMKCDEY") paramSet = STD192Q_4_LMKCDEY;
                // 256-bit classical
                else if (sec == "STD256") paramSet = STD256;
                else if (sec == "STD256_LMKCDEY") paramSet = STD256_LMKCDEY;
                else if (sec == "STD256_3_LMKCDEY") paramSet = STD256_3_LMKCDEY;
                else if (sec == "STD256_4_LMKCDEY") paramSet = STD256_4_LMKCDEY;
                // 256-bit quantum
                else if (sec == "STD256Q") paramSet = STD256Q;
                else if (sec == "STD256Q_LMKCDEY") paramSet = STD256Q_LMKCDEY;
                else if (sec == "STD256Q_3_LMKCDEY") paramSet = STD256Q_3_LMKCDEY;
                else if (sec == "STD256Q_4_LMKCDEY") paramSet = STD256Q_4_LMKCDEY;
                // Low probability of failure
                else if (sec == "LPF_STD128") paramSet = LPF_STD128;
                else if (sec == "LPF_STD128Q") paramSet = LPF_STD128Q;
                else if (sec == "LPF_STD128_LMKCDEY") paramSet = LPF_STD128_LMKCDEY;
                else if (sec == "LPF_STD128Q_LMKCDEY") paramSet = LPF_STD128Q_LMKCDEY;
            }
            if (json->isMember("method")) {
                std::string m = (*json)["method"].asString();
                methodName = m;
                if (m == "GINX") method = GINX;
                else if (m == "AP") method = AP;
                else if (m == "LMKCDEY") method = LMKCDEY;
            }
        }
        
        auto contextId = FHEManager::instance().createContext(paramSet, method);
        
        Json::Value resp;
        resp["context_id"] = contextId;
        resp["security"] = securityName;
        resp["method"] = methodName;
        callback(HttpResponse::newHttpJsonResponse(resp));
    }

    void generateKeys(const HttpRequestPtr& req,
                      std::function<void(const HttpResponsePtr&)>&& callback)
    {
        auto json = req->getJsonObject();
        if (!json || !json->isMember("context_id")) {
            Json::Value err;
            err["error"] = "context_id required";
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }
        
        std::string contextId = (*json)["context_id"].asString();
        
        if (FHEManager::instance().generateKeys(contextId)) {
            Json::Value resp;
            resp["success"] = true;
            resp["context_id"] = contextId;
            callback(HttpResponse::newHttpJsonResponse(resp));
        } else {
            Json::Value err;
            err["error"] = "Context not found";
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k404NotFound);
            callback(resp);
        }
    }

    void encrypt(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback)
    {
        auto json = req->getJsonObject();
        if (!json || !json->isMember("context_id") || !json->isMember("value")) {
            Json::Value err;
            err["error"] = "context_id and value required";
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }
        
        std::string contextId = (*json)["context_id"].asString();
        bool value = (*json)["value"].asBool();
        std::string error;
        
        auto ctId = FHEManager::instance().encrypt(contextId, value, error);
        
        if (ctId.empty()) {
            Json::Value err;
            err["error"] = error;
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
        } else {
            Json::Value resp;
            resp["ciphertext_id"] = ctId;
            callback(HttpResponse::newHttpJsonResponse(resp));
        }
    }

    void decrypt(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback)
    {
        auto json = req->getJsonObject();
        if (!json || !json->isMember("ciphertext_id")) {
            Json::Value err;
            err["error"] = "ciphertext_id required";
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }
        
        std::string ctId = (*json)["ciphertext_id"].asString();
        bool result;
        std::string error;
        
        if (FHEManager::instance().decrypt(ctId, result, error)) {
            Json::Value resp;
            resp["value"] = result;
            callback(HttpResponse::newHttpJsonResponse(resp));
        } else {
            Json::Value err;
            err["error"] = error;
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
        }
    }

    void evalAnd(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback)
    {
        evalBinaryGate(req, std::move(callback), "and");
    }

    void evalOr(const HttpRequestPtr& req,
                std::function<void(const HttpResponsePtr&)>&& callback)
    {
        evalBinaryGate(req, std::move(callback), "or");
    }

    void evalXor(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback)
    {
        evalBinaryGate(req, std::move(callback), "xor");
    }

    void evalNand(const HttpRequestPtr& req,
                  std::function<void(const HttpResponsePtr&)>&& callback)
    {
        evalBinaryGate(req, std::move(callback), "nand");
    }

    void evalNor(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback)
    {
        evalBinaryGate(req, std::move(callback), "nor");
    }

    void evalXnor(const HttpRequestPtr& req,
                  std::function<void(const HttpResponsePtr&)>&& callback)
    {
        evalBinaryGate(req, std::move(callback), "xnor");
    }

    void evalNot(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback)
    {
        auto json = req->getJsonObject();
        if (!json || !json->isMember("ciphertext_id")) {
            Json::Value err;
            err["error"] = "ciphertext_id required";
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }
        
        std::string ctId = (*json)["ciphertext_id"].asString();
        std::string error;
        
        auto resultId = FHEManager::instance().evalNot(ctId, error);
        
        if (resultId.empty()) {
            Json::Value err;
            err["error"] = error;
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
        } else {
            Json::Value resp;
            resp["result_id"] = resultId;
            callback(HttpResponse::newHttpJsonResponse(resp));
        }
    }

  private:
    void evalBinaryGate(const HttpRequestPtr& req,
                        std::function<void(const HttpResponsePtr&)>&& callback,
                        const std::string& gate)
    {
        auto json = req->getJsonObject();
        if (!json || !json->isMember("ct1_id") || !json->isMember("ct2_id")) {
            Json::Value err;
            err["error"] = "ct1_id and ct2_id required";
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }
        
        std::string ct1Id = (*json)["ct1_id"].asString();
        std::string ct2Id = (*json)["ct2_id"].asString();
        std::string error;
        std::string resultId;
        
        auto& mgr = FHEManager::instance();
        if (gate == "and") resultId = mgr.evalAnd(ct1Id, ct2Id, error);
        else if (gate == "or") resultId = mgr.evalOr(ct1Id, ct2Id, error);
        else if (gate == "xor") resultId = mgr.evalXor(ct1Id, ct2Id, error);
        else if (gate == "nand") resultId = mgr.evalNand(ct1Id, ct2Id, error);
        else if (gate == "nor") resultId = mgr.evalNor(ct1Id, ct2Id, error);
        else if (gate == "xnor") resultId = mgr.evalXnor(ct1Id, ct2Id, error);
        
        if (resultId.empty()) {
            Json::Value err;
            err["error"] = error;
            auto resp = HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(k400BadRequest);
            callback(resp);
        } else {
            Json::Value resp;
            resp["result_id"] = resultId;
            callback(HttpResponse::newHttpJsonResponse(resp));
        }
    }
};

}  // namespace fhe
