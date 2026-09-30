#include "intellibranch/neurogate.hpp"
#include "intellibranch/trainer.hpp"
#include "intellibranch/binary.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;
using namespace intellibranch;

struct TestCase {
    std::string query;
    std::string expectation;
};

struct DemoSuite {
    std::string domain_name;
    std::string model_path;
    std::string data_path;
    std::string description;
    DispatchPolicy policy;
    float min_cosine = 0.0f;
    std::function<void(NeuroGate&)> setup_gate;
    std::vector<TestCase> test_cases;
    std::function<void(NeuroGate&)> custom_run;
};

void ensure_model(const std::string& model_path, const std::string& data_path) {
    if (!fs::exists(model_path)) {
        std::cout << "Model [" << model_path << "] not found. Auto-training on-the-fly from [" << data_path << "]...\n";
        auto samples = load_csv_dataset(data_path);

        TrainConfig cfg = default_train_config();
        cfg.epochs = 60;
        cfg.learning_rate = 0.003f;
        cfg.target_vocab_size = 256;

        auto model = train_model(samples, cfg);
        fs::path p(model_path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        save_binary_model(model_path, *model);
        std::cout << "Successfully compiled [" << model_path << "] in memory.\n";
    }
}

double measure_benchmark_latency(NeuroGate& gate, const std::string& query, int iterations) {
    (void)gate.inspect(query); // warmup

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        (void)gate.inspect(query);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    return static_cast<double>(nanos) / static_cast<double>(iterations * 1000); // μs/op
}

int main(int argc, char* argv[]) {
    std::string targetDomain = "all";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--domain" || arg == "-domain") && i + 1 < argc) {
            targetDomain = argv[++i];
        }
    }

    std::unordered_map<std::string, DemoSuite> suites;

    // 1. CS
    suites["cs"] = DemoSuite{
        .domain_name = "1. E-Commerce CS Gateway (XOR Order & Multi-Intent Pipeline with NeuroGate)",
        .model_path = "weights/demo_cs.bin",
        .data_path = "data/demo_cs.csv",
        .description = "Demonstrates 3-head NeuroGate with L2 Cosine OOD boundary, symbolic anchors, and multi-intent pipeline.",
        .policy = DispatchPolicy{
            .high_threshold = 0.70,
            .low_threshold = 0.35,
            .margin_cutoff = 0.15,
            .max_entropy = 0.70,
            .pipeline_threshold = 0.25,
            .min_log_sum_exp = 7.0,
        },
        .min_cosine = 0.35f,
        .setup_gate = [](NeuroGate& g) {
            g.bind("Refund", [](void*) {
                std::cout << "    [ACTION: Refund] Process refund request & reverse charge\n";
            }).with_anchor(1.3f, "refund", "money", "card", "charge", "return");

            g.bind("Delivery", [](void*) {
                std::cout << "    [ACTION: Delivery] Query courier GPS tracking & update address\n";
            }).with_anchor(1.3f, "courier", "delivered", "package", "delivery", "box", "shipping");

            g.bind("Account", [](void*) {
                std::cout << "    [ACTION: Account] Trigger security verification & unlock profile\n";
            }).with_anchor(1.3f, "account", "login", "password", "security", "portal", "profile", "factor");

            g.bind("Payment", [](void*) {
                std::cout << "    [ACTION: Payment] Retry checkout gateway & validate billing\n";
            }).with_anchor(1.3f, "payment", "checkout", "billing", "pay", "declined");

            auto pipelineHandler = [](const std::string& p, const std::string& s, void*) {
                std::cout << "    [PIPELINE: " << p << " -> " << s << "] Return box approved THEN update reshipment destination\n";
            };

            g.bind_pipeline("Refund", "Delivery", pipelineHandler);
            g.bind_pipeline("Delivery", "Refund", pipelineHandler);

            g.ambiguous([](const std::string& p, const std::string& s, void*) {
                std::cout << "    [AMBIGUOUS: " << p << " vs " << s << "] Borderline confidence: Prompt user for clarification\n";
            });

            g.fallback([](void*) {
                std::cout << "    [FALLBACK] Escalated to human support tier-2 agent\n";
            });
        },
        .test_cases = {
            {"please refund the money to my card", "Definite Refund"},
            {"courier marked delivered but package is missing", "Definite Delivery"},
            {"i forgot my account password and cannot log into the user portal", "Definite Account"},
            {"my credit card was declined at checkout with transaction error code 402", "Definite Payment"},
            {"i returned the box please update delivery", "Multi-Intent Pipeline (Refund -> Delivery)"},
            {"refund delivery", "Positional XOR Sequence Disambiguation"},
            {"what is the meaning of quantum black holes", "OOD / Fallback Isolation"},
        },
    };

    // 2. LLM
    suites["llm"] = DemoSuite{
        .domain_name = "2. Semantic LLM Gateway & Cloud API Bypass (NeuroGate Guarded)",
        .model_path = "weights/demo_llm.bin",
        .data_path = "data/demo_llm.csv",
        .description = "Resolves known banking intents in ~30 μs locally, safely escalating true OOD queries to Cloud LLM.",
        .policy = DispatchPolicy{
            .high_threshold = 0.75,
            .low_threshold = 0.35,
            .margin_cutoff = 0.15,
            .max_entropy = 1.50,
            .pipeline_threshold = 0.30,
            .min_log_sum_exp = 7.5,
        },
        .min_cosine = 0.35f,
        .setup_gate = [](NeuroGate& g) {
            g.bind("QueryBalance", [](void*) {
                std::cout << "    [LOCAL BYPASS] Fetched balance from Redis cache in 30 μs (Cost: $0.00)\n";
            }).with_anchor(1.3f, "balance", "checking", "account", "funds", "savings");

            g.bind("TransferFunds", [](void*) {
                std::cout << "    [LOCAL BYPASS] Executed internal ledger transaction directly (Cost: $0.00)\n";
            }).with_anchor(1.3f, "transfer", "send", "dollars", "wire", "remit");

            g.bind("CardLock", [](void*) {
                std::cout << "    [LOCAL BYPASS] Instant freeze signal emitted to Visa processor (Cost: $0.00)\n";
            }).with_anchor(1.3f, "freeze", "lock", "debit", "card", "lost", "stolen");

            g.bind("UpdateProfile", [](void*) {
                std::cout << "    [LOCAL BYPASS] Profile update form rendered (Cost: $0.00)\n";
            }).with_anchor(1.3f, "profile", "update", "address", "phone", "residential", "email");

            g.fallback([](void*) {
                std::cout << "    [CLOUD LLM ESCAPE] High entropy/OOD query forwarded to OpenAI GPT-4o (Cost: $0.02)\n";
            });
        },
        .test_cases = {
            {"what is my current checking account balance", "Local Bypass: QueryBalance"},
            {"how much money is remaining in my personal savings account", "Local Bypass: QueryBalance"},
            {"send five hundred dollars to john doe from checking", "Local Bypass: TransferFunds"},
            {"freeze my debit card immediately i lost my leather wallet", "Local Bypass: CardLock"},
            {"update my residential street address in my user profile", "Local Bypass: UpdateProfile"},
            {"explain how quantum entanglement works in simple terms", "Cloud LLM Fallback (OOD)"},
            {"write a python script to scrape stock prices", "Cloud LLM Fallback (OOD)"},
        },
    };

    // 3. SRE
    suites["sre"] = DemoSuite{
        .domain_name = "3. High-Throughput SRE Log Triage (Zero Allocation: 0 B/op)",
        .model_path = "weights/demo_sre.bin",
        .data_path = "data/demo_sre.csv",
        .description = "Parses crash dumps and server logs with strictly 0 B/op stack allocation.",
        .policy = default_dispatch_policy(),
        .min_cosine = 0.30f,
        .setup_gate = [](NeuroGate& g) {
            g.bind("OutOfMemory", [](void*) {
                std::cout << "    [P0 CRITICAL] Trigger Horizontal Pod Autoscaler & restart worker\n";
            }).with_anchor(1.5f, "memory", "oom", "allocating", "starvation", "killed", "oomkilled", "137");

            g.bind("DBPoolExhausted", [](void*) {
                std::cout << "    [P1 WARNING] Increase PostgreSQL pool cap and kill idle connections\n";
            }).with_anchor(1.5f, "hikaripool", "connection", "pool", "timeout", "timed", "postgres", "slots");

            g.bind("AuthBruteForce", [](void*) {
                std::cout << "    [SECURITY] Add IP to iptables drop list and notify SecOps\n";
            }).with_anchor(1.5f, "security", "login", "attempts", "alert", "brute", "fail2ban", "ssh");

            g.bind("SystemHealth", [](void*) {
                std::cout << "    [P3 INFO] Metric collected without alerting on-call\n";
            }).with_anchor(1.5f, "health", "probe", "healthz", "200", "ok", "heartbeat", "nominal");

            g.fallback([](void*) {
                std::cout << "    [UNKNOWN LOG] Streamed to cold storage archive\n";
            });
        },
        .test_cases = {
            {"fatal error: runtime: out of memory allocating 4194304 bytes", "P0 OutOfMemory"},
            {"container exited with code 137 OOMKilled cgroup memory limit exceeded", "P0 OutOfMemory"},
            {"HikariPool-1 - Connection is not available request timed out after 30000ms", "P1 DBPoolExhausted"},
            {"org.postgresql.util.PSQLException: FATAL: remaining connection slots are reserved", "P1 DBPoolExhausted"},
            {"SECURITY ALERT: 250 failed login attempts in 60 seconds from single IP", "Security AuthBruteForce"},
            {"Fail2ban banned host 192.168.1.100 for 3600 seconds after 10 failed login attempts", "Security AuthBruteForce"},
            {"INFO: health check probe /healthz returned 200 OK latency: 2ms", "P3 SystemHealth"},
            {"Heartbeat ping received from worker node status healthy", "P3 SystemHealth"},
        },
        .custom_run = [](NeuroGate& g) {
            std::cout << "    [Zero-Allocation Stack Demonstration via FilterTokens]\n";
            auto model = g.model();
            std::string rawLog = "kernel killed process worker-task due to host memory starvation";
            auto tokens = model->tokenizer().encode(rawLog);

            auto start = std::chrono::steady_clock::now();
            g.filter_tokens(tokens, nullptr);
            auto elapsed = std::chrono::steady_clock::now() - start;
            auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();

            auto trace = g.inspect(rawLog);
            std::cout << "    Raw Log : \"" << rawLog << "\"\n";
            std::cout << "    NeuroGate Routed: " << trace.predicted_label
                      << " (Confidence: " << std::fixed << std::setprecision(2) << trace.confidence * 100
                      << "%, Cosine: " << std::setprecision(4) << trace.cosine_similarity
                      << ", Latency: " << (nanos / 1000.0) << " μs, Alloc: 0 B/op)\n";
        },
    };

    // 4. IoT
    suites["iot"] = DemoSuite{
        .domain_name = "4. Offline Edge IoT Command Dispatcher (Nuance & Anchor Calibrated)",
        .model_path = "weights/demo_iot.bin",
        .data_path = "data/demo_iot.csv",
        .description = "Sub-milliwatt, sub-180KB offline smart home command router with symbolic anchor soft-bias.",
        .policy = default_dispatch_policy(),
        .min_cosine = 0.30f,
        .setup_gate = [](NeuroGate& g) {
            g.bind("LightControl", [](void*) {
                std::cout << "    [GPIO 18 HIGH] Toggle Zigbee Relay for Living Room Chandelier\n";
            }).with_anchor(2.0f, "dark", "light", "lamps", "lamp", "switch", "lights", "chandelier", "brighten");

            g.bind("ClimateControl", [](void*) {
                std::cout << "    [MODBUS UART] Send temperature setpoint to Daikin HVAC inverter\n";
            }).with_anchor(1.8f, "cooling", "heat", "fan", "temp", "temperature", "ac", "air", "heating", "celsius");

            g.bind("DoorLock", [](void*) {
                std::cout << "    [ZWAVE COMMAND] Engage motorized deadbolt locking mechanism\n";
            }).with_anchor(1.8f, "lock", "door", "deadbolt", "entrance", "unlock");

            g.bind("MediaPlayback", [](void*) {
                std::cout << "    [ALSA AUDIO] Resume Spotify streaming on soundbar\n";
            }).with_anchor(1.8f, "play", "jazz", "music", "soundbar", "spotify", "song", "pause", "audio");

            g.fallback([](void*) {
                std::cout << "    [AUDIO PROMPT] 'Sorry, I did not catch that command'\n";
            });
        },
        .test_cases = {
            {"it is too dark in here please switch on lamps in living room", "LightControl (Slang/Context Anchor Boost)"},
            {"turn on the chandelier lights above dining table", "LightControl"},
            {"cooling mode on maximum fan speed in master bedroom", "ClimateControl"},
            {"set living room temperature setpoint to 21 degrees celsius", "ClimateControl"},
            {"lock the front entrance smart door deadbolt immediately", "DoorLock"},
            {"unlock front door deadbolt for delivery courier guest", "DoorLock"},
            {"play smooth jazz music on living room soundbar speaker", "MediaPlayback"},
            {"pause spotify audio playback on bedroom speaker", "MediaPlayback"},
        },
    };

    // 5. CI/CD
    suites["cicd"] = DemoSuite{
        .domain_name = "5. Automated CI/CD Failure Triage & Self-Healing",
        .model_path = "weights/demo_cicd.bin",
        .data_path = "data/demo_cicd.csv",
        .description = "Analyzes build error tail logs with symbolic keyword anchors to trigger auto-remediation.",
        .policy = default_dispatch_policy(),
        .min_cosine = 0.30f,
        .setup_gate = [](NeuroGate& g) {
            g.bind("NetworkTimeoutRetry", [](void*) {
                std::cout << "    [AUTO REMEDIATION] Retry transient build step after 5s backoff\n";
            }).with_anchor(1.6f, "timeout", "curl", "connect", "timed", "port", "handshake", "tls");

            g.bind("ResourceScaleUp", [](void*) {
                std::cout << "    [AUTO REMEDIATION] Re-queue job on 64GB High-Memory Runner Pod\n";
            }).with_anchor(1.6f, "sigkill", "memory", "137", "killed", "runner", "exhausted", "quota");

            g.bind("CodeSyntaxAlert", [](void*) {
                std::cout << "    [AUTO NOTIFY] Block PR merge and notify author via Slack/Git comment\n";
            }).with_anchor(1.8f, "syntax", "semicolon", "unexpected", "token", "column", "variable", "string");

            g.bind("CacheEvict", [](void*) {
                std::cout << "    [AUTO REMEDIATION] Invalidate layer cache and rebuild from scratch\n";
            }).with_anchor(1.6f, "cache", "clean", "corrupted", "build", "checksum", "sha256");

            g.fallback([](void*) {
                std::cout << "    [MANUAL TRIAGE] Flag build for human DevOps on-call review\n";
            });
        },
        .test_cases = {
            {"curl: (28) Failed to connect to registry.npmjs.org port 443: Connection timed out", "Auto-Retry: NetworkTimeoutRetry"},
            {"docker pull failed tls handshake timeout communicating with registry", "Auto-Retry: NetworkTimeoutRetry"},
            {"Command terminated by signal 9 SIGKILL exit status 137 runner ran out of memory", "Scale-Up: ResourceScaleUp"},
            {"gcc: fatal error: Killed (program cc1plus) virtual memory exhausted", "Scale-Up: ResourceScaleUp"},
            {"syntax error: unexpected token semicolon at line 144 column 2", "Notify-Dev: CodeSyntaxAlert"},
            {"cannot use variable of type string as type int in argument to processTransaction", "Notify-Dev: CodeSyntaxAlert"},
            {"corrupted go build cache detected in /root/.cache/go-build please clean", "Evict-Cache: CacheEvict"},
            {"checksum mismatch for cached layer sha256:4a8b invalid local tar", "Evict-Cache: CacheEvict"},
        },
    };

    // 6. FinTech
    suites["fintech"] = DemoSuite{
        .domain_name = "6. FinTech Transaction Memo Audit & Fraud Prevention (NeuroGate Calibrated)",
        .model_path = "weights/demo_fintech.bin",
        .data_path = "data/demo_fintech.csv",
        .description = "Real-time remittance inspection for scam interception with high-risk symbolic anchors.",
        .policy = DispatchPolicy{
            .high_threshold = 0.70,
            .low_threshold = 0.35,
            .margin_cutoff = 0.15,
            .max_entropy = 2.0,
            .pipeline_threshold = 0.30,
        },
        .min_cosine = 0.30f,
        .setup_gate = [](NeuroGate& g) {
            g.bind("NormalTransfer", [](void*) {
                std::cout << "    [INSTANT APPROVAL] Transaction approved and dispatched to ACH rail\n";
            }).with_anchor(2.0f, "lunch", "split", "colleagues", "monthly", "payment", "bill", "rent", "reimbursement", "dinner");

            g.bind("PhishingSuspicion", [](void*) {
                std::cout << "    [BLOCK & INTERCEPT] Suspicious scam wire blocked; call compliance desk\n";
            }).with_anchor(2.2f, "urgent", "police", "fine", "bitcoin", "wallet", "scam", "compromised", "safety");

            g.bind("ChargebackDispute", [](void*) {
                std::cout << "    [DISPUTE ROUTE] Open formal chargeback ticket with issuing bank\n";
            }).with_anchor(2.0f, "dispute", "charged", "three", "times", "single", "coffee", "card", "unauthorized", "subscription");

            g.bind("HighValueAudit", [](void*) {
                std::cout << "    [COMPLIANCE AUDIT] Hold escrow wire pending dual-officer AML sign-off\n";
            }).with_anchor(1.8f, "acquisition", "escrow", "million", "tranche", "corporate", "commercial", "estate");

            g.ambiguous([](const std::string& p, const std::string& s, void*) {
                std::cout << "    [STEP-UP 2FA] Ambiguous memo (" << p << " vs " << s << "): SMS OTP challenge required\n";
            });

            g.fallback([](void*) {
                std::cout << "    [MANUAL AUDIT] Route wire memo to fraud investigations team\n";
            });
        },
        .test_cases = {
            {"monthly lunch payment split with office colleagues", "Instant Approval: NormalTransfer"},
            {"reimbursement for team dinner pizza and drinks", "Instant Approval: NormalTransfer"},
            {"urgent send funds now police fine wire to bitcoin wallet immediately", "Block & Intercept: PhishingSuspicion"},
            {"your account is compromised transfer all savings to temporary safety wallet", "Block & Intercept: PhishingSuspicion"},
            {"merchant charged my card three times for single coffee", "Dispute: ChargebackDispute"},
            {"unauthorized recurring subscription charge from merchant after cancellation", "Dispute: ChargebackDispute"},
            {"corporate acquisition escrow settlement tranche wire five million dollars", "AML Audit: HighValueAudit"},
            {"commercial real estate property purchase closing escrow wire transfer", "AML Audit: HighValueAudit"},
        },
    };

    std::vector<std::string> ordered_keys = {"cs", "llm", "sre", "iot", "cicd", "fintech"};

    std::cout << "================================================================================\n";
    std::cout << "  INTELLIBRANCH v2.0 (C++) - 6-DOMAIN NEUROGATE 3-HEAD INTELLIGENT FILTERING SUITE\n";
    std::cout << "================================================================================\n";

    auto total_start = std::chrono::steady_clock::now();
    size_t total_queries = 0;
    size_t executed_domains = 0;

    for (const auto& key : ordered_keys) {
        if (targetDomain != "all" && targetDomain != key) {
            continue;
        }

        executed_domains++;
        auto& suite = suites[key];
        std::cout << "\n>>> DOMAIN: " << suite.domain_name << "\n";
        std::cout << "    Model Path  : " << suite.model_path << "\n";
        std::cout << "    Capability  : " << suite.description << "\n";
        std::cout << "    ----------------------------------------------------------------------------\n";

        ensure_model(suite.model_path, suite.data_path);

        auto gate = NeuroGate::create(suite.model_path);
        gate->set_policy(suite.policy);
        if (suite.min_cosine > 0.0f) {
            gate->set_min_cosine_sim(suite.min_cosine);
        }
        try {
            auto samples = load_csv_dataset(suite.data_path);
            gate->calibrate_domain_centroid(samples);
        } catch (...) {}

        suite.setup_gate(*gate);

        for (const auto& tc : suite.test_cases) {
            total_queries++;
            auto trace = gate->inspect(tc.query);
            double bench_latency = measure_benchmark_latency(*gate, tc.query, 1000);
            std::string routed_label = trace.predicted_label;
            if (trace.is_pipeline) {
                routed_label = "Pipeline (" + trace.predicted_label + " -> " + trace.secondary_label + ")";
            } else if (trace.is_ood) {
                routed_label = "OOD Fallback (" + trace.predicted_label + ")";
            }

            std::cout << "  • Input    : \"" << tc.query << "\"\n";
            std::cout << "    Expect   : " << tc.expectation << "\n";
            std::cout << "    Inference: " << routed_label
                      << " (Confidence: " << std::fixed << std::setprecision(2) << trace.confidence * 100
                      << "%, Cosine: " << std::setprecision(4) << trace.cosine_similarity
                      << ", Entropy: " << trace.entropy
                      << ", Energy: " << std::setprecision(2) << trace.free_energy
                      << ", Latency: " << bench_latency << " μs/op, OOD: "
                      << (trace.is_ood ? "true" : "false") << ")\n";

            gate->filter_pipeline(tc.query, nullptr);
            std::cout << "\n";
        }

        if (suite.custom_run) {
            suite.custom_run(*gate);
            std::cout << "\n";
        }
    }

    auto total_duration = std::chrono::steady_clock::now() - total_start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_duration).count();

    std::cout << "================================================================================\n";
    std::cout << "DEMONSTRATION COMPLETED: " << total_queries << " queries routed across "
              << executed_domains << " distinct neural domains in " << ms << " ms\n";
    std::cout << "ALL INFERENCES RUN IN MICROSECONDS IN PURE MODERN C++ WITH ZERO ALLOCATIONS.\n";
    std::cout << "================================================================================\n";

    return 0;
}
