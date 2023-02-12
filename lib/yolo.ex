defmodule Yolo do

  def load_from_pt(path) do
    dict = Bumblebee.Conversion.PyTorch.Loader.load!(path)
    model_dict = Map.fetch!(dict, "model")
    model_yaml = Map.fetch!(model_dict.state, "yaml")

    model = Yolo.Impl.make(model_yaml)
    params = Yolo.Loader.load_pt_data(dict)

    {params, model}
  end

  def load_image(path) do
    v_image = Image.open!(path, access: :random)

    {:ok, image} = Image.to_nx(v_image, shape: :hwc)
    image =
      image
      |> Nx.transpose(axes: [:channels, :height, :width])
      |> Nx.new_axis(0, :batch)
      |> Nx.divide(255)

    image
  end

end
