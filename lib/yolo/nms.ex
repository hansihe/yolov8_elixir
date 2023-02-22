defmodule Yolo.NMS do

  def nms(boxes, prob_thresh \\ 0.8, iou_thresh \\ 0.8) do
    {_anchors, data} = Nx.shape(boxes)

    (0..(data - 5))
    |> Enum.map(fn idx ->
      probs =
        boxes
        |> Nx.slice_along_axis(4 + idx, 1, axis: 1)
        |> Nx.reshape({:auto})

      argsort = Nx.argsort(probs, direction: :desc)

      boxes_ordered = Nx.take(Nx.slice_along_axis(boxes, 0, 4, axis: 1), argsort)
      probs_ordered = Nx.new_axis(Nx.take(probs, argsort), 1)

      concated = Nx.concatenate([boxes_ordered, probs_ordered], axis: 1)

      above_thresh =
        concated
        |> Nx.to_batched(1)
        |> Stream.map(&Nx.to_flat_list/1)
        |> Enum.take_while(fn [_, _, _, _, prob] -> prob > prob_thresh end)

      do_nms(above_thresh, [], iou_thresh)
    end)
  end

  def do_nms([], results, _iou_thresh), do: results

  def do_nms([box1 | rest], results, iou_thresh) do
    rest =
      rest
      |> Stream.map(fn box2 -> {box2, iou(box1, box2)} end)
      |> Stream.reject(fn {_box2, iou} -> iou > iou_thresh end)
      |> Enum.map(fn {bbox2, _iou} -> bbox2 end)

    do_nms(rest, [box1 | results], iou_thresh)
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
