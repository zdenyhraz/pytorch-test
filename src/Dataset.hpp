torch::Tensor TestFunction(const torch::Tensor& x)
{
  return torch::exp(-10. * torch::pow(x - 0.5, 2));
}

class Dataset : public torch::data::Dataset<Dataset>
{
public:
  Dataset()
  {
    mInputs = torch::linspace(0, 1, 11);
    mOutputs = TestFunction(mInputs);
  }

  torch::data::Example<> get(size_t index) override { return {mInputs[index], mOutputs[index]}; }

  torch::optional<size_t> size() const override
  {
    //.sizes() array of dimsizes
    //.size(dim) size of dim
    return mInputs.size(0);
  }

private:
  torch::Tensor mInputs, mOutputs;
};
