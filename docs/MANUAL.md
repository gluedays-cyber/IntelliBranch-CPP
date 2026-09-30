# IntelliBranch-C++: Embedded Neural AI Manual & Tutorial for C++ Developers

This guide provides C++ systems engineers with a technical manual and hands-on tutorial for **IntelliBranch-C++: An Engine That Directly Creates and Runs Its Own Domain Artificial Intelligence**. Learn how to design domain knowledge, generate lightweight neural networks from scratch in seconds, and execute microsecond AI-driven control flow with zero external dependencies in modern C++20.

---

## Table of Contents

1. [Architectural Mental Model for C++ Engineers](#1-architectural-mental-model-for-c-engineers)
   - [Two Phases: Offline Compilation vs. In-Memory Routing](#11-two-phases-offline-compilation-vs-in-memory-routing)
2. [The 4-Step Operational Workflow](#2-the-4-step-operational-workflow)
3. [Keyword & API Reference Manual](#3-keyword--api-reference-manual)
   - [Constructor: `Router::create`](#31-routercreate)
   - [3-Tier Criteria: `DispatchPolicy` & `set_policy`](#32-dispatchpolicy--set_policy)
   - [Branch Binding: `bind`](#33-bind)
   - [Borderline Safety: `ambiguous`](#34-ambiguous)
   - [Multi-Intent: `bind_pipeline` & `default_pipeline`](#35-bind_pipeline--default_pipeline)
   - [Safety Isolation: `fallback`](#36-fallback)
   - [Inference & Branching: `dispatch` & `dispatch_pipeline`](#37-dispatch--dispatch_pipeline)
   - [Whitebox Observability: `inspect` & `RouteTrace`](#38-inspect--routetrace)
   - [Atomic Hot-Swap: `reload` & `swap_model`](#39-reload--swap_model)
   - [Active Learning: `enable_telemetry` & `drain_telemetry`](#310-enable_telemetry--drain_telemetry)
   - [Zero Allocations: `predict_slots`](#311-predict_slots-zero-allocation-inference)
   - [NeuroGate 3-Head Engine: `NeuroGate::create`](#312-neurogate-3-head-geometric-intelligent-filter-engine)
4. [End-to-End Production Tutorial](#4-end-to-end-production-tutorial)
   - [Step 1: AI Design — Structuring Domain Knowledge (`dataset.csv`)](#step-1-ai-design--structuring-domain-knowledge-datasetcsv)
   - [Step 2: Building Your Own AI — Training & Model Generation (`ib-train`)](#step-2-building-your-own-ai--training--model-generation-ib-train)
   - [Step 3: AI-Powered Branching — Microsecond Live Routing (`dispatch`)](#step-3-ai-powered-branching--microsecond-live-routing-dispatch)
5. [Advanced Production Recipes](#5-advanced-production-recipes)
   - [Cancellation Token Propagation](#51-cancellation-token-propagation)
   - [Atomic Zero-Downtime Weight Hot-Reloading](#52-atomic-zero-downtime-weight-hot-reloading)
   - [Whitebox Telemetry & Active Learning Feedback Loop](#53-whitebox-telemetry--active-learning-feedback-loop)
   - [Semantic LLM Gateway & Cloud Bypass](#54-semantic-llm-gateway--cloud-bypass)
   - [Hierarchical Cascading Multi-Router](#55-hierarchical-cascading-multi-router)
6. [Low-Level C++ Runtime Internals](#6-low-level-c-runtime-internals)
   - [6.1. Memory Allocation Breakdown (Stack-Level 0 B/op Inference)](#61-memory-allocation-breakdown-stack-level-0-bop-inference)
   - [6.2. Lock-Free Read Path & Concurrency Guarantees](#62-lock-free-read-path--concurrency-guarantees)
   - [6.3. IBRN Binary Wire Format Specification (Format v2 Positional)](#63-ibrn-binary-wire-format-specification)
   - [6.4. Hardware Cache Locality: Flat 1D Memory vs Pointer Indirection](#64-hardware-cache-locality-flat-1d-memory-vs-pointer-indirection)
7. [6-Domain Multi-Task Demonstration Suite (CLI Guide)](#7-6-domain-multi-task-demonstration-suite-cli-guide)

---

## 1. Architectural Mental Model for C++ Engineers

In standard C++, control flow branching over strings relies on discrete equality or regex checking:

```cpp
// Standard C++: Discrete String Equality
if (input == "refund") {
    return process_refund();
}
```

This works if and only if `input` precisely equals `"refund"`. If the caller sends `"refnd"`, `"I need my money back"`, or `"reverse charge"`, the statement falls through.

**IntelliBranch-C++** replaces discrete string comparisons with **continuous vector coordinate proximity**:

```text
[ Input Text ] ("can u refund order #49281")
     │
     ▼
[ BPE Tokenizer ] ────── Splits into subwords (e.g. "ref", "und") -> immune to typos
     │
     ▼
[ 64-D Latent Coordinates ] ── Similar business intents map to adjacent coordinates
     │
     ▼
[ 128-D GELU Layer ] ── Evaluates context combinations (distinguishes "cancel order" from "cancel alerts")
     │
     ▼
[ Softmax Distribution ] ── Converts scores into probabilities (Refund: 0.98, Delivery: 0.01)
     │
     ▼
[ Branch Dispatch ] ───── Directly executes bound C++ callback in ~30 microseconds
```

### 1.1. Two Phases: Offline Compilation vs. In-Memory Routing

IntelliBranch-C++ divides work into two distinct phases:

| Dimension | Phase 1: Model Compilation (Offline Training) | Phase 2: Router Dispatch (Live In-Memory Inference) |
| :--- | :--- | :--- |
| **Action** | Reads `dataset.csv` and builds compact weights in **1.5 seconds** | Loads `.bin` weights into RAM and routes requests in **30 microseconds** |
| **Output / Result** | A single portable binary file (`intent.bin`, < 150 KB) | Immediate execution of your C++ handler (`router->bind(...)`) |
| **Runtime Resource** | Run once during CI/CD build or server bootstrap | Consumes < 150 KB RAM and **0% background CPU** when idle |

---

## 2. The 4-Step Operational Workflow

1. **AI Design (`dataset.csv`)**: Define discrete business labels (`Refund`, `Delivery`, `Account`) and map 5–10 representative user sentences to each.
2. **AI Compilation (`ib-train`)**: Run the self-contained training pipeline to generate the little-endian binary model.
3. **In-Memory Loading (`Router::create`)**: Construct an immutable routing engine loaded once during bootstrap.
4. **Branch Dispatch (`dispatch`)**: Route real-time user queries through a 3-tier safety gate (`Definite`, `Ambiguous`, `Fallback`).

---

## 3. Keyword & API Reference Manual

### 3.1. `Router::create`

```cpp
std::shared_ptr<Router> Router::create(const std::string& model_path, double default_threshold = 0.60);
```

Loads a binary model file into memory once and constructs an immutable routing core.

### 3.2. `DispatchPolicy` & `set_policy`

Configures the 3-tier confidence criteria:

```cpp
struct DispatchPolicy {
    double high_threshold = 0.75;      // Min confidence for definite route
    double low_threshold = 0.40;       // Threshold below which requests are isolated to fallback
    double margin_cutoff = 0.15;       // Required margin gap between Top-1 and Top-2
    double max_entropy = 2.0;          // Max allowable Shannon entropy before OOD fallback
    double pipeline_threshold = 0.30;  // Secondary confidence for multi-intent pipeline
    double min_log_sum_exp = 0.0;      // Log-Sum-Exp energy boundary
};
```

### 3.3. `bind`

```cpp
Router& bind(const std::string& label, RouteAction action);
```

Binds a business handler to a classification label.

### 3.4. `ambiguous`

```cpp
Router& ambiguous(AmbiguousAction action);
```

Invoked when confidence is borderline or the margin between Top-1 and Top-2 predictions is below `margin_cutoff`.

### 3.5. `bind_pipeline` & `default_pipeline`

```cpp
Router& bind_pipeline(const std::string& primary, const std::string& secondary, PipelineAction action);
Router& default_pipeline(PipelineAction action);
```

Executes when both primary and secondary intents satisfy the multi-intent criteria.

### 3.6. `fallback`

```cpp
Router& fallback(RouteAction action);
```

Registers the safety action for low-confidence, high-entropy, or out-of-domain requests.

### 3.7. `dispatch` & `dispatch_pipeline`

```cpp
void dispatch(const std::string& text, void* payload = nullptr, std::stop_token stop_token = {});
void dispatch_pipeline(const std::string& text, void* payload = nullptr, std::stop_token stop_token = {});
```

Evaluates input text through the neural network and invokes the target route callback.

### 3.8. `inspect` & `RouteTrace`

```cpp
RouteTrace inspect(const std::string& text) const;
```

Generates diagnostic telemetry explaining the routing decision: token IDs, subwords, UNK ratio, probability distribution, confidence, entropy, and latency.

### 3.9. `reload` & `swap_model`

```cpp
void reload(const std::string& model_path);
void swap_model(std::shared_ptr<InferenceModel> new_model);
```

Hot-swaps weights atomically using `std::shared_mutex` without blocking active traffic.

### 3.10. `enable_telemetry` & `drain_telemetry`

```cpp
Router& enable_telemetry(size_t capacity = 1024);
std::vector<TelemetryEvent> drain_telemetry();
```

Maintains an asynchronous circular ring buffer recording ambiguous and fallback queries for active retraining.

### 3.11. `predict_slots` (Zero Allocation Inference)

```cpp
StaticInferenceResult predict_slots(const std::vector<uint32_t>& token_ids, float temperature = 0.0f) const;
```

Calculates Top-2 ranked predictions and entropy on the stack with strictly zero heap allocations.

### 3.12. `NeuroGate::create`

```cpp
auto gate = NeuroGate::create("weights/demo_cs.bin");
gate->bind("Refund", handler).with_anchor(1.3f, "refund", "card", "money");
gate->set_min_cosine_sim(0.35f);
gate->filter_pipeline(query);
```

Combines geometric manifold boundary checks (Head 1), 64-bit keyword bitmask soft-biasing (Head 2), and stack softmax / entropy gating (Head 3).

---

## 4. End-to-End Production Tutorial

### Step 1: AI Design — Structuring Domain Knowledge (`data/sample_dataset.csv`)

```csv
text,label
i want a refund on my purchase,Refund
cancel payment and return my funds,Refund
when will my delivery package arrive,Delivery
track shipment status courier,Delivery
forgot my login account password,Account
reset account credentials security,Account
```

### Step 2: Building Your Own AI (`ib-train`)

```bash
./bin/ib-train --data data/sample_dataset.csv --out weights/intent.bin --epochs 60 --vocab 250 --lr 0.005
```

### Step 3: AI-Powered Branching (`apps/main.cpp`)

```cpp
#include "intellibranch/router.hpp"
#include <iostream>

using namespace intellibranch;

int main() {
    auto router = Router::create("weights/intent.bin", 0.60);

    router->
        bind("Refund", [](void*) { std::cout << "Handling refund\n"; }).
        bind("Delivery", [](void*) { std::cout << "Tracking delivery\n"; }).
        bind("Account", [](void*) { std::cout << "Securing account\n"; }).
        fallback([](void*) { std::cout << "Escalating query\n"; });

    router->dispatch("driver dumped the package in the rain");
    return 0;
}
```

---

## 5. Advanced Production Recipes

### 5.1. Cancellation Token Propagation

```cpp
std::stop_source source;
// ... cancel if request deadline is exceeded
router->dispatch(query, nullptr, source.get_token());
```

### 5.2. Atomic Zero-Downtime Weight Hot-Reloading

```cpp
// Background worker atomically swaps model weights
std::thread reloader([&]() {
    router->reload("weights/intent_v2.bin");
});
reloader.detach();
```

### 5.3. Whitebox Telemetry & Active Learning Feedback Loop

```cpp
router->enable_telemetry(2048);

// Periodically drain recorded edge cases to retrain dataset
auto feedback = router->drain_telemetry();
for (const auto& ev : feedback) {
    if (ev.is_fallback) {
        log_drift_candidate(ev.input_text, ev.confidence);
    }
}
```

---

## 6. Low-Level C++ Runtime Internals

### 6.1. Memory Allocation Breakdown (Stack-Level 0 B/op Inference)

By utilizing fixed-size `std::array` buffers for pooling and logits up to `MAX_GATE_CLASSES` (16) and `MAX_GATE_EMB_DIM` (64), forward evaluations inside `NeuroGate::evaluate_fast` and `InferenceModel::predict_slots` execute completely on the CPU stack cache without invoking `malloc` or heap managers.

### 6.2. Lock-Free Read Path & Concurrency Guarantees

Model weights and vocabulary mappings are completely immutable after loading. Thread safety for concurrent reader threads is achieved through `std::shared_lock<std::shared_mutex>`, allowing hundreds of parallel reader threads to evaluate requests simultaneously without lock contention.

### 6.3. IBRN Binary Wire Format Specification

```text
[0..3]   Magic Bytes: 'I', 'B', 'R', 'N' (4 bytes)
[4..7]   Format Version: uint32 LE (1 or 2)
[8..11]  VocabSize: uint32 LE
[12..15] EmbeddingDim: uint32 LE
[16..19] HiddenDim: uint32 LE
[20..23] NumClasses: uint32 LE
[24..]   Labels & Vocabulary string blocks
[...]    Merge Rules block (token1, token2, target)
[...]    IEEE-754 float32 weight tensors: Embedding, Positional (v2), W1, B1, W2, B2
[End-32] SHA-256 Checksum (32 bytes)
```

### 6.4. Hardware Cache Locality: Flat 1D Memory vs Pointer Indirection

All tensor weights are stored in contiguous 1D memory blocks (`std::vector<float>`). Dense linear projections execute over stride-1 memory chunks, maximizing CPU L1/L2 cache line hits and enabling automatic SIMD vectorization (AVX2/NEON).

---

## 7. 6-Domain Multi-Task Demonstration Suite (CLI Guide)

```bash
# Run all demonstration suites
./bin/ib-demo --domain all

# Run specific domain suites
./bin/ib-demo --domain cs      # E-Commerce CS Gateway
./bin/ib-demo --domain llm     # Semantic LLM Gateway & Cloud Bypass
./bin/ib-demo --domain sre     # High-Throughput SRE Log Triage
./bin/ib-demo --domain iot     # Offline Edge IoT Command Dispatcher
./bin/ib-demo --domain cicd    # Automated CI/CD Failure Triage
./bin/ib-demo --domain fintech # FinTech Transaction Memo Audit
```
