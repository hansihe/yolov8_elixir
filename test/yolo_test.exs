defmodule YoloTest do
  use ExUnit.Case
  doctest Yolo

  def make_from_file(path) do
    dict = Bumblebee.Conversion.PyTorch.Loader.load!(path)
    model_dict = dict["model"]

    model_root_keys = Map.keys(model_dict.state)
    model_yaml = model_dict.state["yaml"]

    Yolo.Impl.make(model_yaml)
  end

  @tag :skip
  test "greets the world" do
    path = "/home/hansihe/proj/ml/pricetag_yolov8_sparse/best.pt"
    model = make_from_file(path)

    bbox_model = Yolo.Util.process_output(model)
    #%{"image" => image_dims} = Axon.get_inputs(abc)
    #image_template = Nx.template(image_dims, :f32)
    #IO.inspect(Axon.get_output_shape(abc, %{"image" => image_template}))

    #Yolo.Util.dist_to_bbox(model.outputs, model.reg_box, model.feat_dims, model.strides)

    {init_fn, predict_fn} = Axon.build(bbox_model, compiler: EXLA)

    %{"image" => image_dims} = Axon.get_inputs(bbox_model)
    image_template = Nx.template(image_dims, :f32)

    #Axon.Display.as_table(model, image_template)
    #|> IO.puts()

    params = init_fn.(image_template, %{})
    load_params = Yolo.Loader.load_pt(path)

    params
    |> Map.keys()
    |> Enum.sort()
    |> Enum.filter(fn name -> String.contains?(name, "22.detect.") end)
    |> Enum.each(fn name ->
      IO.puts(name)
      #IO.inspect(params[name])
    end)

    IO.puts("==================")

    load_params
    |> Map.keys()
    |> Enum.sort()
    |> Enum.filter(fn name -> String.contains?(name, "22.detect.") end)
    |> Enum.each(fn name ->
      IO.puts(name)
      #IO.inspect(load_params[name])
    end)

    image = Nx.broadcast(0.5, image_dims)
    result = predict_fn.(load_params, image)

    boxes_raw =
      result
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.reshape({:auto, 5})

    boxes =
      boxes_raw
      |> Yolo.NMS.nms()

    IO.inspect(Enum.count(boxes), label: :num_detections)
    IO.inspect(result)


    #IO.inspect(result)
  end

  @tag :skip
  test "test infer" do
    path = "/home/hansihe/proj/ml/pricetag_yolov8_sparse/best.pt"
    model = make_from_file(path)

    bbox_model = Yolo.Util.process_output(model)

    {init_fn, predict_fn} = Axon.build(bbox_model, compiler: EXLA)

    load_params = Yolo.Loader.load_pt(path)

    v_image = Image.open!("/home/hansihe/Downloads/aaaaa.jpg", access: :random)

    {:ok, image} = Image.to_nx(v_image, shape: :hwc)
    image =
      image
      |> Nx.transpose(axes: [:channels, :height, :width])
      |> Nx.new_axis(0, :batch)
      |> Nx.divide(255)

    #image = Nx.broadcast(0.5, image_dims)
    result = predict_fn.(load_params, image)

    boxes_raw =
      result
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.reshape({:auto, 5})

    boxes =
      boxes_raw
      |> Yolo.NMS.nms()

    IO.inspect(Enum.count(boxes), label: :num_detections)
    IO.inspect(boxes)
  end

  test "test infer2" do
    path = "/home/hansihe/proj/ml/pricetag_yolov8_sparse/best.pt"
    {params, model} = Yolo.load_from_pt(path)

    bbox_model = Yolo.Util.process_output(model)
    {init_fn, predict_fn} = Axon.build(bbox_model, compiler: EXLA, debug: true)

    image = Yolo.load_image("/home/hansihe/Downloads/aaaaa.jpg")

    template = Nx.template({1, 3, 640, 640}, :f32)
    dev_params = init_fn.(%{"image" => template}, params)
    result = predict_fn.(dev_params, %{"image" => image})

    boxes_raw =
      result
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.reshape({:auto, 5})

    boxes =
      boxes_raw
      |> Yolo.NMS.nms()

    IO.inspect(Enum.count(boxes), label: :num_detections)
    IO.inspect(boxes)
  end

end
