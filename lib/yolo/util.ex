defmodule Yolo.Util do
  import Nx.Defn

  def process_output(model) do
    %{
      outputs: outputs,
      reg_max: reg_max,
      num_classes: nc,
      feat_dims: feat_dims,
      stride: stride,
    } = model

    {dist, cls} = Axon.split(outputs, [reg_max * 4, nc], axis: 1)

    cls = Axon.sigmoid(cls)
    bboxes = dist_to_bbox(dist, reg_max, feat_dims, stride)

    Axon.concatenate(bboxes, cls, axis: 1)
  end

  defn conv(x, kernel, _opts \\ []) do
    Nx.conv(x, kernel)
  end

  def make_anchors(feat_dims, stride) do
    # Make anchor offset and stride tensors.
    #
    # Anchor offsets are the grid of center points of the spatial locations
    # in the feature vectors. These simply increment by one in a grid for
    # each of the feature vector layers.
    #
    # Anchor strides will transform spatial coordinates from the scale of
    # the feature tensor to the scale of the input image by multiplication.
    #
    # Example:
    #  * An image has dimensions 640x640
    #  * Feature vectors will be of dimensions:
    #    * P3 -> 80x80
    #    * P4 -> 40x40
    #    * P5 -> 20x20
    #  * Once offset values are calculated, they will need to be converted
    #    to absolute coordinates. This is done by adding and subtracting
    #    them from the anchor points.
    #  * The absolute coordinates are still in the scale of the feature
    #    vector, and will need to be converted to the scale of the image.
    #    This is done by multiplying by the scale difference:
    #    * P3 -> 640 / 80 = 8
    #    * P4 -> 640 / 40 = 16
    #    * P5 -> 640 / 20 = 32

    offsets =
      Enum.map(feat_dims, fn [s1, s2] ->
        [
          Nx.add(Nx.iota({s1, s2}, axis: 1), 0.5),
          Nx.add(Nx.iota({s1, s2}, axis: 0), 0.5),
        ]
        |> Nx.stack()
        |> Nx.reshape({1, 2, :auto})
      end)
      |> Nx.concatenate(axis: 2)

    strides =
      Enum.map(Enum.zip(stride, feat_dims), fn {stride, [s1, s2]} ->
        Nx.broadcast(stride, {1, s1 * s2})
      end)
      |> Nx.concatenate(axis: 1)

    {offsets, strides}
  end

  def dist_to_bbox(dist, reg_max, feat_dims, stride) do
    {offsets_tensor, strides_tensor} = make_anchors(feat_dims, stride)
    offsets = Axon.constant(offsets_tensor)
    strides = Axon.constant(strides_tensor)

    # Make the kernel that transforms the discretized offset probability
    # distribution into a single offset value.
    # There are `reg_max` bins in the discretized probability distribution.
    dist_values =
      Nx.iota({1, reg_max, 1, 1})
      |> Axon.constant()

    # Prepare values for convolution with the kernel.
    dist_weights =
      dist
      |> Axon.reshape({:batch, 4, reg_max, :auto})
      |> Axon.transpose([0, 2, 1, 3])
      |> Axon.softmax(axis: 1)

    # Perform the convolution.
    # This will reduce the reg_max bins into a single offset value.
    # There are now 4 offsets for each bbox (l,t,r,b).
    {tl_off, br_off} =
      Axon.layer(&conv/3, [dist_weights, dist_values])
      |> Axon.reshape({:batch, 4, :auto})
      # Split offsets into tl and rb
      |> Axon.split([2, 2], axis: 1)

    # Convert offsets into absolute coordinates.
    tl = Axon.subtract(offsets, tl_off)
    br = Axon.add(offsets, br_off)

    # Convert corner coordinates into center,size format.
    center = Axon.multiply(Axon.add(br, tl), Axon.constant(Nx.tensor(0.5)))
    size = Axon.subtract(br, tl)

    # Concat into final output tensor.
    Axon.concatenate(center, size, axis: 1)
    # Multiply by the feature tensor stride in order to get correctly
    # scaled output coordinates.
    |> Axon.multiply(strides)
  end

end
