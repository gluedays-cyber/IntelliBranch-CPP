#include "intellibranch/trainer.hpp"
#include "intellibranch/binary.hpp"
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace intellibranch;

int main(int argc, char* argv[]) {
    std::string datasetPath = "data/sample_dataset.csv";
    std::string outputPath = "weights/model.bin";
    size_t epochs = 150;
    size_t targetVocab = 150;
    float lr = 0.005f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--data" || arg == "-data") && i + 1 < argc) {
            datasetPath = argv[++i];
        } else if ((arg == "--out" || arg == "-out") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if ((arg == "--epochs" || arg == "-epochs") && i + 1 < argc) {
            epochs = std::stoul(argv[++i]);
        } else if ((arg == "--vocab" || arg == "-vocab") && i + 1 < argc) {
            targetVocab = std::stoul(argv[++i]);
        } else if ((arg == "--lr" || arg == "-lr") && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        }
    }

    std::cout << "Loading dataset from: " << datasetPath << "\n";
    auto samples = load_csv_dataset(datasetPath);
    std::cout << "Loaded " << samples.size() << " training samples\n";

    TrainConfig cfg = default_train_config();
    cfg.epochs = epochs;
    cfg.target_vocab_size = targetVocab;
    cfg.learning_rate = lr;
    cfg.batch_size = 16;
    cfg.patience = 10;

    std::cout << "Starting offline BPE + AdamW training pipeline...\n";
    auto model = train_model(samples, cfg);

    fs::path outPath(outputPath);
    if (outPath.has_parent_path()) {
        fs::create_directories(outPath.parent_path());
    }

    std::cout << "Serializing trained model to Little-Endian binary: " << outputPath << "\n";
    save_binary_model(outputPath, *model);

    std::cout << "Training and binary export completed successfully.\n";
    std::cout << "Model saved at: " << outputPath
              << " (Vocab: " << model->header().vocab_size
              << ", Classes: " << model->header().num_classes << ")\n";

    return 0;
}
