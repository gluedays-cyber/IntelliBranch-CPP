#include "intellibranch/router.hpp"
#include "intellibranch/trainer.hpp"
#include "intellibranch/binary.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace intellibranch;

// 1. Business Logic Handlers
void handleRefund(void* payload) {
    auto* q = static_cast<std::string*>(payload);
    std::cout << "[ACTION: Refund]   Processing refund for: '" << (q ? *q : "") << "'\n";
}

void handleDelivery(void* payload) {
    auto* q = static_cast<std::string*>(payload);
    std::cout << "[ACTION: Delivery] Querying shipment tracking for: '" << (q ? *q : "") << "'\n";
}

void handleAccount(void* payload) {
    auto* q = static_cast<std::string*>(payload);
    std::cout << "[ACTION: Account]  Initiating account security for: '" << (q ? *q : "") << "'\n";
}

void handleFallback(void* payload) {
    auto* q = static_cast<std::string*>(payload);
    std::cout << "[FALLBACK: Safety] Isolated low-confidence request: '" << (q ? *q : "") << "'\n";
}

int main() {
    std::string modelPath = "weights/intent.bin";

    // Auto-compile model if missing (ensures instant zero-config clone & run)
    if (!fs::exists(modelPath)) {
        std::cout << "Model weights not found. Compiling from data/sample_dataset.csv...\n";
        auto samples = load_csv_dataset("data/sample_dataset.csv");
        TrainConfig cfg = default_train_config();
        cfg.epochs = 50;
        cfg.learning_rate = 0.005f;
        cfg.target_vocab_size = 250;

        auto model = train_model(samples, cfg);
        fs::create_directories("weights");
        save_binary_model(modelPath, *model);
        std::cout << "Model compilation completed.\n";
    }

    // 2. Load compiled binary weights into memory (0.60 calibrated threshold)
    auto router = Router::create(modelPath, 0.60);

    // 3. Bind routes directly inside main
    router->
        bind("Refund", handleRefund).
        bind("Delivery", handleDelivery).
        bind("Account", handleAccount).
        fallback(handleFallback);

    // 4. Execute microsecond branch dispatch
    std::vector<std::string> testQueries = {
        "I want to cancel my payment and request a refund",
        "When will my delivery package arrive",
        "Forgot my account password",
        "Please refund my purchase",
        "Track my shipment status",
        "Completely random gibberish noise 12345!@#$",
        "hey where is my stuff it was supposed to get here yesterday",
        "can u cancel order #49281? i bought it by mistake",
        "bruh the reset link is not sending to my email, fix this",
        "got charged twice on my card, refund the extra charge asap",
        "item arrived totally smashed, want my money back",
        "cant log into my acct keeps saying wrong password",
        "tracking says delivered but nothing is in my mailbox",
        "yo i typed the wrong apt number, can someone update the address before it ships",
        "sent the return box a week ago, when do i get my refund?",
        "locked out of my account after 3 tries... help pls",
        "ordered a large but you guys sent me a small",
        "any update on order #88412? hasnt moved in 4 days",
        "how do i just delete my account permanently? done with this site",
        "driver dumped the package in the rain, everything inside is ruined",
        "promo code didnt apply at checkout, can u refund the difference",
        "need a real person, this bot is completely useless",
        "can i change the delivery date? nobody will be home this friday",
        "my card was charged but never received any confirmation email or receipt",
        "lost access to my 2FA phone number, how do i get back in",
        "package has been stuck in transit for 10 days straight, is it lost or what",
    };

    std::cout << "=== IntelliBranch Server Routing Started ===\n";
    for (auto& query : testQueries) {
        try {
            router->dispatch(query, &query);
        } catch (const std::exception& e) {
            std::cerr << "Dispatch error: " << e.what() << "\n";
        }
    }
    std::cout << "=== All queries dispatched in microseconds ===\n";
    return 0;
}
