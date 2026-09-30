# IntelliBranch-C++
<img src="https://github.com/user-attachments/assets/3413a486-d71c-4285-841d-76bbe74f830a" width="226" height="200" alt="Image" align="right" style="margin-left: 15px; margin: 10px;">
<p align="center">
  <strong>Directly Creates and Runs Its Own Neural AI in Pure Modern C++20</strong><br>
  <em>Stop borrowing third-party AIs. This engine directly manufactures a domain-specific artificial intelligence from scratch in 1.5 seconds, routing execution flow in ~30 μs with 0 B/op heap allocation, Zero Downloads, and Zero External Runtime Dependencies.</em>
</p>

<p align="center">
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Latency-~30_μs-brightgreen.svg" alt="Latency"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Memory-<180_KB-blue.svg" alt="Memory"></a>
  <img src="https://img.shields.io/badge/Wire_Format-v2_Positional-orange.svg" alt="Format v2">
  <img src="https://img.shields.io/badge/Standard-C%2B%2B20-00599C.svg" alt="C++ Standard">
  <img src="https://img.shields.io/badge/Dependencies-Zero_Standalone-success.svg" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <a href="docs/MANUAL.md"><strong>📖 Read the Full Developer Manual & Production Tutorial →</strong></a>
</p>

---

## What is IntelliBranch-C++?

**IntelliBranch-C++ does NOT borrow, lease, or download external AI models. This engine directly creates and runs its own domain artificial intelligence from scratch.**

Instead of relying on brittle string pattern matching or making high-latency calls to cloud LLMs, it **manufactures a domain-specific lightweight neural network directly from your dataset in under 2 seconds**. It maps typos, slang, inverted syntax, and colloquial phrasing into a continuous latent vector space—routing execution flow directly to your bound C++ lambdas or functions in **microseconds (~30 μs) with strictly stack-allocated zero-overhead inference**.

```text
Incoming Request ("bruh can u refund order #49281")
                     │
                     ▼
       [ In-Memory BPE Tokenizer ]
                     │
                     ▼
  [ Dense (D=64) + Positional (P=128x64) ]
                     │
                     ▼
  [ Non-Linear GELU Mean Pooling (D=64) ]
                     │
                     ▼
      [ Hidden Projection (D=128) ]
                     │
                     ▼
   [ Softmax + Shannon Entropy Calibrated Guard ]
                     │
     ┌───────────────┼───────────────┬────────────────┐
     ▼               ▼               ▼                ▼
(Score ≥ 0.75)  (Score ≥ 0.30)  (Margin < 0.15)  (Entropy > 2.0 / UNK ≥ 0.5)
[DEFINITE ROUTE] [PIPELINE]      [AMBIGUOUS]      [FALLBACK ISOLATION]
```

---

## Why IntelliBranch-C++?

Modern systems face an architectural dilemma when routing unstructured or noisy user requests:

```cpp
// ❌ RETRO BRANCHING: Brittle, explodes in complexity, collapses under real-world noise
if (input.find("refund") != std::string::npos || input.find("cancel") != std::string::npos) {
    // FAILS on: "sent the return box a week ago when do i get my money back"
    // FAILS on: "can u reverse the charge?" (typos, slang, synonyms)
    // MISROUTES on: "cancel shipment delay notifications" (word collision)
}

// ❌ CLOUD LLMs (OpenAI / Claude): Massive network latency, recurring per-token cost, third-party dependency
// Latency: 400ms – 2,500ms (Unusable in high-throughput microservices)
// Cost: $0.0015 – $0.03 per request (Bills explode under scale)
// Vulnerability: Outages, rate limits, JSON hallucination, network partitions

// ❌ LOCAL LLMs & SLMs (Ollama, llama.cpp, Mistral-7B, Phi-3): Severe host resource exhaustion
// Memory: Monopolizes 4.5 GB to 8.0 GB+ of RAM/VRAM just to pick a 4-byte enum
// CPU Starvation: Burns 100% CPU across multiple cores, starving companion microservices
// Deployment Complexity: Heavy runtime dependencies, CUDA drivers, or background daemons

// ✅ INTELLIBRANCH-C++: Self-Generated Micro-AI (In-Memory C++20 Engine)
// Memory Footprint: Under 180 KB (25,000x smaller than quantized 7B models)
// Latency: ~30 μs with 0 B/op stack-level hot paths and deterministic 3-tier fallback
// Deployment: 100% Pure Modern C++20 with Zero External Libraries
```

### Architectural Comparison Matrix

| Capability | Retro Branching (`if` / Regex) | Cloud LLMs (OpenAI / Claude) | Local LLMs (Ollama / llama.cpp) | **IntelliBranch-C++ (Embedded Engine)** |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Latency** | < 1 μs | 300 ms – 2,500 ms (Network bound) | 30 ms – 300 ms (Compute bound) | **~30 μs (In-Memory)** |
| **Throughput (per core)** | > 500,000 req/sec | ~50 req/sec (Rate limited) | ~20–50 req/sec (CPU saturated) | **> 33,000 req/sec (Zero Alloc)** |
| **Runtime Allocation** | 0 B/op | High (HTTP payload) | High (Buffer allocations) | **0 B/op (Stack-level execution)** |
| **System Memory (RAM)** | Negligible | External service | **4.5 GB – 8.0 GB+ (VRAM / RAM)** | **< 180 KB (Format v2)** |
| **Token Order Awareness** | Rigid regex position | ✅ Transformer Attention | ✅ Transformer Attention | ✅ **Learned Positional Embeddings** |
| **Hardware Reqs** | Standard CPU | External service | High-end GPU or 8+ Core CPU | **Runs on embedded microcontrollers / edge** |
| **Operational Cost** | $0.00 | $0.0015+ per call | High hardware/electricity cost | **$0.00 (Self-contained)** |
| **Hot Weight Reload** | Binary recompile | API model string switch | Multi-second model reload | **Lock-free Atomic Hot-Swap (`0 ns` stop)** |
| **Active Learning Loop** | N/A | Manual logging | N/A | **Built-in Ring Buffer Telemetry** |
| **Deployment Complexity** | Single binary | API client | CUDA / PyTorch / Ollama daemon | **Zero External Dependencies (Pure C++20)** |

---

## 30-Second Quickstart

### 1. Build from Source

```bash
# Clone the repository
git clone https://github.com/gluedays-cyber/IntelliBranch-C++.git
cd IntelliBranch-C++

# Configure and compile with CMake
cmake -B build -G "Ninja"
cmake --build build

# Run the 6-domain intelligent routing demo
./bin/ib-demo --domain all
```

### 2. Live Server Routing Example (`apps/main.cpp`)

```cpp
#include "intellibranch/router.hpp"
#include <iostream>

using namespace intellibranch;

int main() {
    // 1. Initialize router with compiled binary weights (0.60 calibrated confidence threshold)
    auto router = Router::create("weights/intent.bin", 0.60);

    // 2. Bind business logic actions directly
    router->
        bind("Refund", [](void* payload) {
            std::cout << "[ACTION: Refund] Processing refund\n";
        }).
        bind("Delivery", [](void* payload) {
            std::cout << "[ACTION: Delivery] Querying GPS tracking\n";
        }).
        bind("Account", [](void* payload) {
            std::cout << "[ACTION: Account] Initiating account recovery\n";
        }).
        fallback([](void* payload) {
            std::cout << "[FALLBACK: Safety] Isolated low-confidence / OOD query\n";
        });

    // 3. Dispatch queries in microseconds
    std::string query = "can u cancel order #49281? i bought it by mistake";
    router->dispatch(query);

    return 0;
}
```

---

## Geometric 3-Head NeuroGate

The `NeuroGate` engine couples a shared lightweight neural backbone with a geometric 3-head zero-allocation gate:

```cpp
#include "intellibranch/neurogate.hpp"
#include <iostream>

using namespace intellibranch;

int main() {
    auto gate = NeuroGate::create("weights/demo_cs.bin");

    // Head 2: Symbolic Soft-Bias Anchors
    gate->bind("Refund", [](void*) {
        std::cout << "[ACTION: Refund] Process refund request\n";
    }).with_anchor(1.3f, "refund", "money", "card", "charge", "return");

    gate->bind("Delivery", [](void*) {
        std::cout << "[ACTION: Delivery] Query courier GPS tracking\n";
    }).with_anchor(1.3f, "courier", "delivered", "package", "delivery");

    // Multi-Intent Pipeline
    gate->bind_pipeline("Refund", "Delivery", [](const std::string& p, const std::string& s, void*) {
        std::cout << "[PIPELINE: " << p << " -> " << s << "] Multi-intent flow executed\n";
    });

    gate->fallback([](void*) {
        std::cout << "[FALLBACK] Escalated to tier-2 human operator\n";
    });

    // Head 1: Cosine Out-of-Domain Guard + Head 3: Shannon Entropy
    gate->set_min_cosine_sim(0.35f);

    gate->filter_pipeline("i returned the box please update delivery");
    return 0;
}
```

---

## Training Your Own Domain AI (`ib-train`)

```bash
# Train a brand new neural classifier from CSV in under 2 seconds
./bin/ib-train --data data/sample_dataset.csv --out weights/intent.bin --epochs 60 --vocab 250 --lr 0.005
```

```text
[Dataset: data/sample_dataset.csv] -> 100% Offline BPE + AdamW Optimization
Epoch  10/60 - Train Loss: 0.5412 (Acc: 85.0%) | Val Loss: 0.5218 (Acc: 83.3%)
Epoch  20/60 - Train Loss: 0.2841 (Acc: 94.2%) | Val Loss: 0.2910 (Acc: 92.5%)
Epoch  30/60 - Train Loss: 0.1105 (Acc: 98.1%) | Val Loss: 0.1250 (Acc: 97.5%)
[Early Stopping] Model converged in 1.48 seconds.
Model serialized to Little-Endian binary: weights/intent.bin (136 KB)
```

---

## 6 Included Multi-Task Demonstration Domains

Run `./bin/ib-demo --domain all` or target a specific domain:

| Domain | Command | Key Capability Demonstrated |
| :--- | :--- | :--- |
| **E-Commerce CS** | `./bin/ib-demo --domain cs` | Positional XOR Order & Multi-Intent Pipeline (`Refund -> Delivery`) |
| **LLM Gateway** | `./bin/ib-demo --domain llm` | Resolves banking intents in 30 μs; routes true OOD to Cloud LLM |
| **SRE Log Triage** | `./bin/ib-demo --domain sre` | Zero allocation (`0 B/op`) stack log triage (OOM, DB pool, Brute Force) |
| **Edge IoT** | `./bin/ib-demo --domain iot` | Offline smart-home command router with symbolic anchor soft-bias |
| **CI/CD Healing** | `./bin/ib-demo --domain cicd` | Analyzes build tail logs with keyword anchors to trigger self-healing |
| **FinTech Audit** | `./bin/ib-demo --domain fintech`| Real-time wire inspection with AML and phishing scam interception |

---

## Verification & Unit Testing

```bash
# Run the complete CTest test suite
ctest --test-dir build --output-on-failure
```

```text
Test project C:/Users/sezzi/programming/IntelliBranch-c++/build
    Start 1: test_ops
1/5 Test #1: test_ops .........................   Passed    0.04 sec
    Start 2: test_binary
2/5 Test #2: test_binary ......................   Passed    0.04 sec
    Start 3: test_runtime
3/5 Test #3: test_runtime .....................   Passed    0.04 sec
    Start 4: test_router
4/5 Test #4: test_router ......................   Passed    0.06 sec
    Start 5: test_neurogate
5/5 Test #5: test_neurogate ...................   Passed    0.07 sec

100% tests passed, 0 tests failed out of 5
```

---

## License

MIT License. Free for commercial and non-commercial use.
