struct Net : torch::nn::Module
{
  static constexpr size_t kInputSize = 1;
  static constexpr size_t kOutputSize = 1;

  Net()
  {
    fc1 = register_module("fc1", torch::nn::Linear(kInputSize, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 16));
    fc4 = register_module("fc4", torch::nn::Linear(16, kOutputSize));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    // x = torch::relu(fc1->forward(x.reshape({x.size(0), kInputSize})));
    // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    // x = torch::log_softmax(fc3->forward(x), /*dim=*/1);

    x = torch::relu(fc1->forward(x.reshape({x.size(0), kInputSize})));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(fc3->forward(x));
    x = fc4->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};