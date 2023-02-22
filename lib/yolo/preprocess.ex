defmodule Yolo.Preprocess do
  @channel_counts [1, 3]

  defmodule ScalingConfig do
    @moduledoc """
    Scale is always applied before translation.

    This means translation coordinates are in the target
    coordinate frame.
    """

    defstruct [
      source_size: {640, 640},
      target_size: {640, 640},
      scale: {1.0, 1.0},
      translation: {0.0, 0.0}
    ]

  end

  @doc """
  Performs the following operations on the input image(s):
  * Converts the image to float, scales values appropriately.
  * Letterboxes image to fit aspect ratio of model input.
  * Resizes image to model input.

  Will return a tuple of `{processed_image, scaling_config}`.
  `scaling_config` can later be used to transform coordinates
  into the source image coordinate frame.
  """
  def preprocess(image, {ch, width, height} \\ {3, 640, 640}) do
    {source_h, source_w, had_batch, transposed} = case Nx.shape(image) do
      {^ch, h, w} ->
        {h, w, false, Nx.new_axis(image, 0)}

      {_, ^ch, h, w} ->
        {h, w, true, image}

      {h, w, ^ch} ->
        image =
          image
          |> Nx.new_axis(0)
          |> Nx.transpose(axes: [0, 3, 1, 2])
        {h, w, false, image}

      {_, h, w, ^ch} ->
        image =
          image
          |> Nx.transpose(axes: [0, 3, 1, 2])
        {h, w, true, image}
    end

    width_ratio = width / source_w
    height_ratio = height / source_h
    ratio = min(width_ratio, height_ratio)

    {scaled_width, scaled_height} =
      if width_ratio < height_ratio do
        {width, ceil(source_h * ratio)}
      else
        {ceil(source_w * ratio), height}
      end

    float = case Nx.type(transposed) do
      {:f, 32} -> transposed
      {:u, 8} -> Nx.divide(Nx.as_type(transposed, :f32), 255)
    end

    scaled = Axon.Layers.resize(float, size: {scaled_height, scaled_width}, channels: :first)

    width_padding = (width - scaled_width) / 2
    height_padding = (height - scaled_height) / 2

    padding_config = [
      {0, 0, 0},
      {0, 0, 0},
      {floor(height_padding), ceil(height_padding), 0},
      {floor(width_padding), ceil(width_padding), 0}
    ]
    padded = Nx.pad(scaled, 0.0, padding_config)

    {_, ^ch, ^width, ^height} = Nx.shape(padded)

    config = %ScalingConfig{
      source_size: {source_w, source_h},
      target_size: {width, height},
      scale: {ratio, ratio},
      translation: {floor(width_padding), floor(height_padding)}
    }

    if had_batch do
      {padded, config}
    else
      {padded[0], config}
    end
  end

end
