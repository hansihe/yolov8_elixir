defmodule Yolo.Loader do

  def load_pt(path) do
    dict = Bumblebee.Conversion.PyTorch.Loader.load!(path)
    load_pt_data(dict)
  end

  def load_pt_data(dict) do
    model_dict = Map.fetch!(dict, "model")
    #model_yaml = Map.fetch!(model_dict.state, "yaml")
    model_modules = Map.fetch!(Map.fetch!(Map.fetch!(model_dict.state, "_modules"), "model").state, "_modules")

    model_modules
    |> Enum.reduce(%{}, fn {idx, obj}, state ->
      idx = String.pad_leading(idx, 2, "0")
      load_layer(state, idx, obj)
    end)
  end

  @noparam_layers ["upsample", "cat"]

  def constructor_to_name("ultralytics.nn.modules.Conv.__new__"), do: "conv"
  def constructor_to_name("torch.nn.modules.upsampling.Upsample.__new__"), do: "upsample"
  def constructor_to_name("ultralytics.nn.modules.Concat.__new__"), do: "cat"
  def constructor_to_name("ultralytics.nn.modules.C2f.__new__"), do: "c2f"
  def constructor_to_name("ultralytics.nn.modules.Detect.__new__"), do: "detect"
  def constructor_to_name("ultralytics.nn.modules.SPPF.__new__"), do: "sppf"
  def constructor_to_name("torch.nn.modules.batchnorm.BatchNorm2d.__new__"), do: "batch_norm"
  def constructor_to_name("torch.nn.modules.conv.Conv2d.__new__"), do: "conv2d"
  def constructor_to_name("torch.nn.modules.container.ModuleList.__new__"), do: "module_list"
  def constructor_to_name("ultralytics.nn.modules.Bottleneck.__new__"), do: "bottle_neck"

  def parameter(nil), do: nil
  def parameter(param) do
    %{
      constructor: "torch._utils._rebuild_parameter",
      args: [tensor | _rest],
    } = param
    tensor
  end

  def maybe_parameter(state, name, param) do
    tensor = parameter(param)
    case tensor do
      nil -> state
      tensor -> Map.put(state, name, tensor)
    end
  end

  def load_layer(state, n, obj = %Unpickler.Object{}) do
    layer_name = constructor_to_name(obj.constructor)
    name = "#{n}"

    load_layer_by_name(state, layer_name, name, obj)
  end

  def load_layer_by_name(state, "batch_norm", name, obj) do
    name = "#{name}.batch_norm"

    %{
      state: %{
        "_buffers" => %{
          "running_mean" => running_mean,
          "running_var" => running_var,
        },
        "_parameters" => %{
          "bias" => bias,
          "weight" => weight,
        },
      },
    } = obj

    data = %{
      "mean" => running_mean,
      "var" => running_var,
      "beta" => parameter(bias),
      "gamma" => parameter(weight),
    }

    Map.put(state, name, data)
  end

  def load_layer_by_name(state, "conv2d", name, obj) do
    name = "#{name}.conv2d"

    %{
      state: %{
        "_parameters" => %{
          "bias" => bias,
          "weight" => weight,
        },
      },
    } = obj

    data = %{
      "kernel" => parameter(weight),
    }
    |> maybe_parameter("bias", bias)

    Map.put(state, name, data)
  end

  def load_layer_by_name(state, "conv", name, obj) do
    name = "#{name}.conv"

    %Unpickler.Object{
      state: %{
        "_modules" => %{
          "conv" => conv,
          "bn" => batch_norm,
        }
      }
    } = obj

    state
    |> load_layer(name, conv)
    |> load_layer(name, batch_norm)
  end

  def load_layer_by_name(state, "c2f", name, obj) do
    name = "#{name}.c2f"

    %{
      state: %{
        "_modules" => %{
          "cv1" => cv1,
          "m" => m,
          "cv2" => cv2,
        },
      },
    } = obj

    state
    |> load_layer("#{name}.cv1", cv1)
    |> load_module_list("#{name}.m", m)
    |> load_layer("#{name}.cv2", cv2)
  end

  def load_layer_by_name(state, "bottle_neck", name, obj) do
    name = "#{name}.bottle_neck"

    %{
      state: %{
        "_modules" => %{
          "cv1" => cv1,
          "cv2" => cv2,
        },
      },
    } = obj

    state
    |> load_layer("#{name}.cv1", cv1)
    |> load_layer("#{name}.cv2", cv2)
  end

  def load_layer_by_name(state, "detect", name, obj) do
    name = "#{name}.detect"

    %{
      state: %{
        "_modules" => %{
          "cv2" => cv2,
          "cv3" => cv3,
        },
      },
    } = obj

    state
    |> container_reduce("#{name}.cv2", cv2, fn state, name, module ->
      load_module_list(state, "#{name}", module)
    end)
    |> container_reduce("#{name}.cv3", cv3, fn state, name, module ->
      load_module_list(state, "#{name}", module)
    end)
  end

  def load_layer_by_name(state, "sppf", name, obj) do
    name = "#{name}.sppf"

    %{
      state: %{
        "_modules" => %{
          "cv1" => cv1,
          "cv2" => cv2,
        },
      },
    } = obj

    state
    |> load_layer("#{name}.cv1", cv1)
    |> load_layer("#{name}.cv2", cv2)
  end

  def load_layer_by_name(state, layer_name, _name, _obj) when layer_name in @noparam_layers do
    state
  end

  def load_module_list(state, layer_name, obj) do
    container_reduce(state, layer_name, obj, fn state, name, module ->
      load_layer(state, name, module)
    end)
  end

  def container_reduce(state, layer_name, container, fun) do
    case container do
      %{
        constructor: "torch.nn.modules.container.ModuleList.__new__",
        state: %{
          "_modules" => modules,
        },
      } ->
        Enum.reduce(modules, state, fn {idx, module}, state ->
          fun.(state, "#{layer_name}.#{idx}", module)
        end)
      %{
        constructor: "torch.nn.modules.container.Sequential.__new__",
        state: %{
          "_modules" => modules,
        },
      } ->
        Enum.reduce(modules, state, fn {idx, module}, state ->
          fun.(state, "#{layer_name}.#{idx}", module)
        end)
    end
  end

end
