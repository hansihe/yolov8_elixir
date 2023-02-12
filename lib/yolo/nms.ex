defmodule Yolo.NMS do

  def nms(boxes) do
    probs =
      boxes
      |> Nx.slice_along_axis(4, 1, axis: 1)
      |> Nx.reshape({:auto})

    boxes_ordered =
      boxes
      |> Nx.take(Nx.argsort(probs, direction: :desc))

    above_thresh =
      boxes_ordered
      |> Nx.to_batched(1)
      |> Stream.map(&Nx.to_flat_list/1)
      |> Enum.take_while(fn [_, _, _, _, prob] -> prob > 0.8 end)

    do_nms(above_thresh, [])
  end

  def do_nms([], results), do: results

  def do_nms([box1 | rest], results) do
    rest =
      rest
      |> Stream.map(fn box2 -> {box2, iou(box1, box2)} end)
      |> Stream.reject(fn {_box2, iou} -> iou > 0.8 end)
      |> Enum.map(fn {bbox2, _iou} -> bbox2 end)

    do_nms(rest, [box1 | results])
  end

  def iou([x1, y1, w1, h1 | _], [x2, y2, w2, h2 | _]) do
    area1 = w1 * h1
    area2 = w2 * h2

    xx = max(x1 - (w1 / 2), x2 - (w2 / 2))
    yy = max(y1 - (h1 / 2), y2 - (h2 / 2))
    aa = min(x1 + (w1 / 2), x2 + (w2 / 2))
    bb = min(y1 + (h2 / 2), y2 + (h2 / 2))

    w = max(0, aa - xx)
    h = max(0, bb - yy)

    intersection_area = w * h

    union_area = area1 + area2 - intersection_area

    intersection_area / union_area
  end

end
