defmodule Yolo.Impl do

  defmodule BuildState do
    defstruct [
      idx: 0,
      layers: %{},
    ]

    def init do
      %__MODULE__{}
    end

    def put_layer(%__MODULE__{} = state, idx, value) do
      %{ state |
        layers: Map.put(state.layers, idx, value)
      }
    end

    def add_layer(%__MODULE__{} = state, value) do
      %{ state |
        idx: state.idx + 1,
        layers:
          state.layers
          |> Map.put(-1, value)
          |> Map.put(state.idx, value),
      }
    end

    def get(%__MODULE__{} = state, idx) do
      Map.fetch!(state.layers, idx)
    end
  end

  def make(spec) do
    state =
      BuildState.init()
      |> BuildState.put_layer(-1, Axon.input("image", shape: {1, 3, 640, 640}))

    #%{"image" => image_dims} = Axon.get_inputs(BuildState.get(state, -1))

    layers_yaml = spec["backbone"] ++ spec["head"]
    state = Enum.reduce(Enum.with_index(layers_yaml), state, fn {layer_spec, idx}, state ->
      idx_str = String.pad_leading("#{idx}", 2, "0")

      name = "#{idx_str}"
      layer = make_layer(name, layer_spec, spec, state)
      BuildState.add_layer(state, layer)
    end)

    BuildState.get(state, -1)
  end

  def p(base, sub), do: base <> "." <> sub

  def make_layer(name, [input_idx, 1, "Conv", [ch, kernel, stride]], spec, state) do
    width_multiple = Map.fetch!(spec, "width_multiple")
    ch = round(ch * width_multiple)

    BuildState.get(state, input_idx)
    |> conv(ch, kernel, stride, 1, name)
  end

  def make_layer(name, [input_idx, base_n, "C2f", [ch]], spec, state) do
    make_layer(name, [input_idx, base_n, "C2f", [ch, false]], spec, state)
  end

  def make_layer(name, [input_idx, base_n, "C2f", [ch, shortcut]], spec, state) do
    name = "#{name}.c2f"

    width_multiple = Map.fetch!(spec, "width_multiple")
    depth_multiple = Map.fetch!(spec, "depth_multiple")
    ch = round(ch * width_multiple)

    n = round(base_n * depth_multiple)
    half_ch = div(ch, 2)

    {top, bot} =
      BuildState.get(state, input_idx)
      |> conv(ch, 1, 1, 0, p(name, "cv1"))
      |> Axon.split(2, axis: 1, name: p(name, "split1"))

    {slices, _prev} =
      Enum.reduce(0..(n-1), {[bot, top], bot}, fn n, {slices, prev} ->
        out = bottleneck(prev, half_ch, shortcut, p(name, "m.#{n}"))
        {[out | slices], out}
      end)

    Enum.reverse(slices)
    |> Axon.concatenate(axis: 1, name: p(name, "cat1"))
    |> conv(ch, 1, 1, 0, p(name, "cv2"))
  end

  def make_layer(name, [input_idx, 1, "SPPF", [ch, 5]], spec, state) do
    name = "#{name}.sppf"

    width_multiple = Map.fetch!(spec, "width_multiple")

    ch = round(ch * width_multiple)
    hidden_ch = div(ch, 2)

    inp =
      BuildState.get(state, input_idx)
      |> conv(hidden_ch, 1, 1, 0, p(name, "cv1"))

    kernel = 5
    padding = div(kernel, 2)
    padding_spec = [{padding, padding}, {padding, padding}]
    l1 = Axon.max_pool(inp, kernel_size: kernel, strides: 1, padding: padding_spec, channels: :first, name: p(name, "maxpool1"))
    l2 = Axon.max_pool(l1, kernel_size: kernel, strides: 1, padding: padding_spec, channels: :first, name: p(name, "maxpool2"))
    l3 = Axon.max_pool(l2, kernel_size: kernel, strides: 1, padding: padding_spec, channels: :first, name: p(name, "maxpool3"))

    Axon.concatenate([inp, l1, l2, l3], axis: 1, name: p(name, "cat1"))
    |> conv(ch, 1, 1, 0, p(name, "cv2"))
  end

  def make_layer(name, [input_idx, 1, "nn.Upsample", ["None", scale, "nearest"]], _spec, state) do
    name = "#{name}.upsample"

    input = BuildState.get(state, input_idx)

    %{"image" => image_dims} = Axon.get_inputs(input)
    {_batches, _ch, d1, d2} = Axon.get_output_shape(input, Nx.template(image_dims, :f32))

    Axon.resize(input, {d1 * scale, d2 * scale}, channels: :first, name: p(name, "resize"))
  end

  def make_layer(name, [input_idxs, 1, "Concat", [1]], _spec, state) do
    name = "#{name}.concat"

    inputs = Enum.map(input_idxs, &BuildState.get(state, &1))
    Axon.concatenate(inputs, axis: 1, name: p(name, "cat"))
  end

  def make_layer(name, [input_idxs, 1, "Detect", [nc_key]], spec, state) do
    name = "#{name}.detect"

    inputs = Enum.map(input_idxs, &BuildState.get(state, &1))
    nc = Map.fetch!(spec, nc_key)

    reg_max = 16
    # The number of outputs per anchor.
    # Consists of (probability distrobution of sides) + (class probability channels)
    num_outputs = (reg_max * 4) + nc

    # Calculate the dimension of each input feature vector level.
    %{"image" => image_dims} = Axon.get_inputs(hd(inputs))
    {_batch, _ch, _s1, ref_s2} = image_dims
    image_template = Nx.template(image_dims, :f32)
    dims = Enum.map(inputs, &Axon.get_output_shape(&1, image_template))
    chs = Enum.map(dims, fn {_batch, ch, _s1, _s2} -> ch end)

    # The stride for each feature level in pixels.
    # Detection heads will be placed evenly on each stride in a grid.
    #
    # Example for yolo8n:
    # * For a 3x640x640 input image, the feature vectors will be of sizes:
    #   * P3 ->  64x80x80
    #   * P4 -> 128x40x40
    #   * P5 -> 256x20x20
    # * For each feature vector, one detection head will be placed on each
    #   slice in the spatial dimension.
    #   * P3 -> 80*80 = 6400 detection heads
    #   * P4 -> 40*40 = 1600 detection heads
    #   * P3 -> 20*20 = 400 detection heads
    #   Making for 8400 total detection heads, with the same number of
    #   candidate output boxes from the model.
    # * Each detection head outputs detections in offsets from the middle
    #   of its location in the feature vector.
    # * The strides are the distances between points in the grid of anchor
    #   points for each detection head in each feature vector level.
    #
    # The calculation here is:
    #   image_spatial_dimension / feature_spatial_dimension = stride
    # for yolov8n at 640x640:
    #   P3 -> 640 / 80 = 8px
    #   P4 -> 640 / 40 = 16px
    #   P5 -> 640 / 20 = 32px
    stride = Enum.map(dims, fn {_batch, _ch, _s1, s2} -> ref_s2 / s2 end)

    c2 = max(div(hd(chs), 4), reg_max * 4)
    c3 = max(hd(chs), nc)

    outputs =
      Enum.map(Enum.with_index(inputs), fn {input, idx} ->
        bbox =
          input
          |> conv(c2, 3, 1, 1, p(name, "cv2.#{idx}.0"))
          |> conv(c2, 3, 1, 1, p(name, "cv2.#{idx}.1"))
          |> Axon.conv(4 * reg_max, channels: :first, name: p(name, "cv2.#{idx}.2.conv2d"))

        classes =
          input
          |> conv(c3, 3, 1, 1, p(name, "cv3.#{idx}.0"))
          |> conv(c3, 3, 1, 1, p(name, "cv3.#{idx}.1"))
          |> Axon.conv(nc, channels: :first, name: p(name, "cv3.#{idx}.2.conv2d"))

        Axon.concatenate(bbox, classes, axis: 1, name: p(name, "#{idx}.cat"))
        |> Axon.reshape({:batch, num_outputs, :auto}, name: p(name, "#{idx}.reshape"))
      end)
      |> Axon.concatenate(axis: -1, name: p(name, "cat"))
      #|> Axon.split([reg_max * 4, nc], axis: 1, name: p(name, "split"))

    %{
      outputs: outputs,
      reg_max: reg_max,
      num_classes: nc,
      #classes: classes,
      #bbox_dists: bbox_dists,
      stride: stride,
      feat_dims: Enum.map(dims, fn {_batch, _ch, s1, s2} -> [s1, s2] end),
    }
  end

  #def make_anchors(feats, strides, grid_cell_offset \\ 0.5) do
  #  anchor_points = Enum.map(strides)
  #end

  def bottleneck(x, ch, true, name) do
    x
    |> bottleneck(ch, false, name)
    |> Axon.add(x, name: p(name, "bottle_neck.add"))
  end

  def bottleneck(x, ch, false, name) do
    name = p(name, "bottle_neck")

    x
    |> conv(div(ch, 2), 3, 1, 1, p(name, "cv1"))
    |> conv(ch, 3, 1, 1, p(name, "cv2"))
  end

  def conv(x, ch, kernel, stride, padding, name) do
    padding_spec = [
      {padding, padding},
      {padding, padding}
    ]

    x
    |> Axon.conv(ch, kernel_size: kernel, strides: stride, padding: padding_spec, use_bias: false, channels: :first, name: p(name, "conv.conv2d"))
    |> Axon.batch_norm(channel_index: 1, name: p(name, "conv.batch_norm"))
    |> Axon.silu(name: p(name, "conv.silu"))
  end

end
