src = File.read!("/Users/ernest.lee/.hermes/cache/scratch/mmg-bridges/tools/bridges.exs")
  |> String.replace("System.halt(Bridges.main())", "")
  |> String.replace("defp embed(", "def embed(") |> String.replace("defp generate(", "def generate(")
Code.eval_string(src, [], file: "/Users/ernest.lee/.hermes/cache/scratch/mmg-bridges/tools/bridges.exs")
s = "/Users/ernest.lee/.hermes/cache/scratch/stillcrawl"
ctx = %{root: "/Users/ernest.lee/Desktop/smoke-water", cache: s <> "/.bridges",
        mmg: "/Users/ernest.lee/.hermes/cache/scratch/mmg-bridges", stamp: "20261025"}
case System.argv() do
  ["tries", tsv] ->
    t = System.monotonic_time(:millisecond)
    IO.inspect(Bridges.embed(ctx, tsv, s <> "/emb") |> elem(0), label: "embed")
    IO.puts("embed ms #{System.monotonic_time(:millisecond) - t}")
    IO.inspect(Bridges.generate(ctx, tsv, s <> "/emb", s <> "/tries"), label: "generate")
    IO.puts("total ms #{System.monotonic_time(:millisecond) - t}")
  steps -> for st <- steps do
    t = System.monotonic_time(:millisecond)
    IO.puts("== #{st}"); IO.inspect(apply(Bridges, String.to_atom("step_" <> st), [ctx]), label: st)
    IO.puts("STEP-MS #{st} #{System.monotonic_time(:millisecond) - t}")
  end
end
