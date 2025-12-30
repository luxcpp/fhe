/**
 *  Lux FHE HTTP/gRPC Server
 *  
 *  This server provides REST and gRPC APIs for FHE operations,
 *  compatible with the Go implementation for comparison testing.
 *
 *  Build:
 *    mkdir build && cd build
 *    cmake .. && make
 *
 *  Run:
 *    ./fhe_server --port 8080
 *
 *  Endpoints:
 *    POST /v1/context/create     - Create FHE context
 *    POST /v1/keys/generate      - Generate keys
 *    POST /v1/encrypt            - Encrypt boolean value
 *    POST /v1/decrypt            - Decrypt ciphertext
 *    POST /v1/eval/and           - AND gate
 *    POST /v1/eval/or            - OR gate
 *    POST /v1/eval/xor           - XOR gate
 *    POST /v1/eval/not           - NOT gate
 *    POST /v1/eval/nand          - NAND gate
 *    GET  /health                - Health check
 */

#include <http/http.h>
#include "fhe_controller.h"

using namespace http;

int main(int argc, char* argv[])
{
    int port = 8080;
    
    // Parse command line args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Lux FHE Server\n"
                      << "Usage: fhe_server [options]\n"
                      << "Options:\n"
                      << "  --port PORT    Listen port (default: 8080)\n"
                      << "  --help         Show this help\n";
            return 0;
        }
    }

    LOG_INFO << "Starting Lux FHE Server on port " << port;
    LOG_INFO << "Endpoints:";
    LOG_INFO << "  POST /v1/context/create";
    LOG_INFO << "  POST /v1/keys/generate";
    LOG_INFO << "  POST /v1/encrypt";
    LOG_INFO << "  POST /v1/decrypt";
    LOG_INFO << "  POST /v1/eval/{and,or,xor,not,nand,nor,xnor}";
    LOG_INFO << "  GET  /health";

    app()
        .setLogLevel(trantor::Logger::kInfo)
        .addListener("0.0.0.0", port)
        .setThreadNum(std::thread::hardware_concurrency())
        .enableGrpc()
        .run();

    return 0;
}
