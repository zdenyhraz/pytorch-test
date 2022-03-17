#include "Precompiled.hpp"
#include "Dataset.hpp"
#include "Net.hpp"

int main(int argc, char** argv)
{
  static constexpr size_t kBatchSize = 1;
  static constexpr size_t kEpochCount = 300;

  fmt::print("Hi bro xd\n");

  // Create a new Net.
  auto net = std::make_shared<Net>();

  // Generate your data set. At this point you can add transforms to you data set, e.g. stack your batches into a single tensor.
  auto dataset = Dataset().map(torch::data::transforms::Stack<>());

  // Generate a data loader.
  auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), kBatchSize);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  for (size_t epochIndex = 0; epochIndex < kEpochCount; ++epochIndex)
  {
    size_t batchIndex = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *dataloader)
    {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(batch.data);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::mse_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 1 batches.
      if (++batchIndex % 1 == 0)
      {
        std::cout << "Epoch: " << epochIndex << " | Batch: " << batchIndex << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        // torch::save(net, "net.pt");
      }
    }
  }

  torch::Tensor x = torch::linspace(0, 1, 11);
  torch::Tensor ytrue = TestFunction(x);
  torch::Tensor ypred = net->forward(x);

  for (int64_t i = 0; i < x.size(0); ++i)
  {
    auto trueval = ytrue[i].item<float>();
    auto predval = ypred[i].item<float>();
    fmt::print("True: {:.2f}, Pred: {:.2f}, Error: {:.2f}\n", trueval, predval, predval - trueval);
  }
}