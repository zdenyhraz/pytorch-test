#include "Precompiled.hpp"
#include "Dataset.hpp"
#include "Net.hpp"

int main(int argc, char** argv)
{
  static constexpr size_t kEpochCount = 500;
  static constexpr size_t kBatchSize = 16; // dataset size has to be divisible by this
  static constexpr float kLearningRate = 0.001;

  static constexpr bool kSaveNetwork = false;
  static constexpr size_t kSaveNetworkCount = 5;
  static constexpr bool kLogProgress = true;
  static constexpr size_t kLogProgressCount = 10;

  auto net = std::make_shared<Net>();

  auto dataset = Dataset(256).map(torch::data::transforms::Stack<>());
  auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), kBatchSize);

  auto datasetTest = Dataset(64).map(torch::data::transforms::Stack<>());
  auto dataloaderTest = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(datasetTest), *datasetTest.size());

  torch::optim::Adam optimizer(net->parameters(), kLearningRate); // SGD, Adam, ...

  for (size_t epochIndex = 0; epochIndex < kEpochCount; ++epochIndex)
  {
    for (auto& batch : *dataloader)
    {
      optimizer.zero_grad();
      torch::Tensor prediction = net->forward(batch.data).reshape({kBatchSize});
      torch::Tensor loss = torch::mse_loss(prediction, batch.target);
      loss.backward();
      optimizer.step();
    }

    if (kLogProgress and epochIndex % (kEpochCount / kLogProgressCount) == 0)
    {
      // evaluate training and test loss on the entire dataset
      torch::NoGradGuard noGrad;

      torch::Tensor prediction = net->forward(dataloader->begin()->data);
      torch::Tensor loss = torch::mse_loss(prediction, dataloader->begin()->target);

      torch::Tensor predictionTest = net->forward(dataloaderTest->begin()->data);
      torch::Tensor lossTest = torch::mse_loss(predictionTest, dataloaderTest->begin()->target);

      fmt::print("Epoch {} | LossTrain {} | LossTest {}\n", epochIndex, loss.item<float>(), lossTest.item<float>());
    }

    if (kSaveNetwork and epochIndex % (kEpochCount / kSaveNetworkCount) == 0)
      torch::save(net, fmt::format("../debug/net_epoch{}.pt", epochIndex));
  }

  torch::Tensor x = torch::linspace(0, 1, 11);
  torch::Tensor ytrue = TestFunction(x);
  torch::Tensor ypred = net->forward(x);

  for (int64_t i = 0; i < x.size(0); ++i)
  {
    auto xval = x[i].item<float>();
    auto trueval = ytrue[i].item<float>();
    auto predval = ypred[i].item<float>();
    fmt::print("x: {:.2f} | True: {:.2f} | Pred: {:.2f} | Error: {:.2f}\n", xval, trueval, predval, predval - trueval);
  }
}