# The join at each end of a bridge clip, picked by the body: body.gd --joins writes
# <out>/joins/<clip>.f32, frames x targets x (max, mean) metres, the face-vertex distance of every
# frame of every clip to each target (a bridge's first or last frame) on the posed GPU body in the
# character frame, over all 18,056 vertices. That is the metric itself per frame, so the argmin
# below is exact. For each target the join is the database frame of the target's kind (all
# database frames when none has it) with the least mean distance, ties broken by the max.
# Controls, per join:
#   neighbours  both frames next to the pick (one frame off either way, inside its clip) are
#               larger by (mean, max)
#   planted     the pick moved one frame must fail the neighbours test (its neighbour is the pick)
#   self        the target against its own frame is 0 (the batched GPU path kept frame order)
# Functions return {:ok, value} or {:error, reason}; nothing raises.
defmodule BridgeJoins do
  def run(out, kinds, bridges) do
    with {:ok, tn} <- File.read(Path.join([out, "joins", "targets.txt"])),
         targets = String.split(tn, "\n", trim: true),
         clips = Path.wildcard(Path.join([out, "joins", "*.f32"])) |> Enum.map(&Path.basename(&1, ".f32")),
         {:ok, data} <- load(clips, &floats(Path.join([out, "joins", &1 <> ".f32"]))),
         {:ok, kd} <- load(clips, &kinds_of(Path.join(kinds, &1 <> ".kinds"))) do
      ntg = length(targets)
      db = clips -- bridges
      nf = fn c -> div(tuple_size(data[c]), ntg * 2) end
      # (mean, max): the order the pick and the controls compare in.
      at = fn c, f, t -> {elem(data[c], (f * ntg + t) * 2 + 1), elem(data[c], (f * ntg + t) * 2)} end

      case Enum.find(clips, &(tuple_size(kd[&1]) != nf.(&1))) do
        nil -> {:ok, targets |> Enum.with_index() |> Enum.map(&join(&1, db, kd, nf, at))}
        c -> {:error, "#{c}: #{tuple_size(kd[c])} kinds for #{nf.(c)} body frames"}
      end
    end
  end

  defp join({name, t}, db, kd, nf, at) do
    [b, fs] = String.split(name, "@")
    fb = String.to_integer(fs)
    kt = elem(kd[b], fb)
    same = for c <- db, f <- 0..(nf.(c) - 1), elem(kd[c], f) == kt, do: {c, f}
    cands = if same == [], do: for(c <- db, f <- 0..(nf.(c) - 1), do: {c, f}), else: same
    {c, f} = Enum.min_by(cands, fn {c, f} -> at.(c, f, t) end)
    d = at.(c, f, t)
    near = fn g -> Enum.filter([g - 1, g + 1], &(&1 >= 0 and &1 < nf.(c))) end
    smaller = fn g -> Enum.reject(near.(g), &(at.(c, &1, t) > at.(c, g, t))) end
    plant = if f + 1 < nf.(c), do: f + 1, else: f - 1
    {self_mean, self_max} = at.(b, fb, t)
    nb_ok = smaller.(f) == []
    plant_fails = smaller.(plant) != []
    self_ok = self_mean == 0.0 and self_max == 0.0
    mm = &:erlang.float_to_binary(&1 * 1000, decimals: 1)
    nbs = Enum.map_join(near.(f), ", ", fn g ->
      {gm, gx} = at.(c, g, t)
      "@#{g} (kind #{elem(kd[c], g)}) mean #{mm.(gm)} max #{mm.(gx)}"
    end)

    %{ok: nb_ok and plant_fails and self_ok, name: name, kind: kt, pick: "#{c}@#{f}", mean: elem(d, 0), max: elem(d, 1),
      cols: [name, kt, "#{c}@#{f}", mm.(elem(d, 1)), mm.(elem(d, 0)), nbs, if(nb_ok, do: "larger", else: "NOT larger"),
             "#{c}@#{plant} " <> if(plant_fails, do: "fails as it must", else: "PASSED (broken)"), mm.(self_max)],
      line: "JOIN #{if fb == 0, do: "into", else: "out of"} #{b} (#{name}, kind #{kt}) <-> #{c}@#{f}: max #{mm.(elem(d, 1))} " <>
            "mean #{mm.(elem(d, 0))} mm; one frame off #{nbs}: #{if nb_ok, do: "larger as it must", else: "NOT larger"}; " <>
            "planted pick #{c}@#{plant}: #{if plant_fails, do: "fails as it must", else: "PASSED (broken)"}; self #{mm.(self_max)} mm"}
  end

  defp load(clips, read) do
    Enum.reduce_while(clips, {:ok, %{}}, fn c, {:ok, m} ->
      case read.(c) do
        {:ok, v} -> {:cont, {:ok, Map.put(m, c, v)}}
        {:error, r} -> {:halt, {:error, "#{c}: #{inspect(r)}"}}
      end
    end)
  end

  defp floats(path) do
    with {:ok, b} <- File.read(path), do: {:ok, for(<<x::float-32-little <- b>>, do: x) |> List.to_tuple()}
  end

  defp kinds_of(path) do
    with {:ok, t} <- File.read(path),
         do: {:ok, t |> String.split("\n", trim: true) |> Enum.map(&String.to_integer/1) |> List.to_tuple()}
  end
end
