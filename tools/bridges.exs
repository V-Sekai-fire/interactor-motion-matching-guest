# Walk-mode bridge clips: Kimodo clips that carry the character between the matcher's kinds
# (stand, crouch, crawl) and into a stop, each drawn as orbit views to check frame by frame.
#   pixi run walk-bridges          (from the smoke-water client root; no arguments)
# Steps, each skipped when its output exists under <root>/.bridges:
#   tools    motion_native (Kimodo's text encoder, host) and mm_kinds (the matcher's per-frame kind)
#   weights  Kimodo-SOMA-RP-v1.1-gguf and the Llama-3-Kimodo text bundle from Hugging Face,
#            checked against the bundle's MANIFEST.json
#   engine   the manifest's 5-repository/godot-double macOS editor, checked against its SHA256SUMS
#   sandbox  godot-sandbox's green Builds run on main (macOS double), in the add-on from
#            4-entities/godot-sandbox
#   guest    motion.elf from interactor-motion-guest's green guest run on main
#   control  the database clip crouch_idle made again on motion.elf must equal the stored one
#            (within 1e-3), so the generated clips come from the same Kimodo as the database
#   embed    tools/bridges/prompts.tsv through the text encoder
#   generate every prompt on motion.elf (Kimodo on ggml-rd, local RenderingDevice), 10 s, seed 1171,
#            50 steps, as the database's clips
#   kinds    mm_kinds: the matcher's per-frame kind for the database and the new clips
#   assets   the SOMA-2603.16858 (SOMA-X) asset release of interactor-soma-x, checked against its
#            SHA256SUMS.txt
#   kernels  contract-anny-kernels' SPIR-V (kernels/anny/gen.sh --no-emit, the committed Slang)
#   body     every clip (new and database) set up once on the CPU (tools/bridges/body_frames.py:
#            static tables, per-frame pose and hips), then posed on the GPU in a windowed Godot
#            (tools/bridges/body.gd: the anny-kernels chain, correctives on); parity on @oracle
#            against SOMA-X's own torch vertices within 1% of the clip's largest corrective, and
#            the correctives-off control must fail it; also the join distances (below) on the GPU
#   joins    the gap at each clip switch into and out of a bridge clip, as mesh face-vertex distance
#            on the posed body in the character frame: max and mean over the 18,056 vertices
#            against the closest database frame of the same kind; the same join one frame off
#            must be larger (control), and each target against itself must be 0
#   sheets   DEFERRED: body orbit views (contract-orbit-views), render_sheets/1
#   video    DEFERRED: one CineForm MKV per clip from the godot-cineform engine, render_video/1
# Tokens: gh's own login (Bao-rotated) and bao-hf-token on stdin; neither reaches argv.
# Functions return {:ok, value} or {:error, reason}; nothing raises.

defmodule Bridges do
  import Bitwise

  @steps ~w(tools weights engine sandbox guest control embed generate kinds assets kernels body joins sheets video)
  @seconds 10
  @seed 1171
  @rows_per_sheet 60
  @kimodo_repo "chibifire/Kimodo-SOMA-RP-v1.1-gguf"
  @kimodo_file "kimodo-soma-rp-v1.1.gguf"
  @text_repo "chibifire/Llama-3-Kimodo-GGML"
  @text_dir "generated/llm2vec-text-bundle"
  @assets_repo "V-Sekai-fire/interactor-soma-x"
  @assets_tag "assets-v0.1.0-dev.1"
  @kernels ~w(soma_orient soma_jump soma_twist soma_expand soma_skin soma_corr_l1 soma_corr_l2 anny_lbs)
  # The two clips checked against SOMA-X's torch vertices: the largest corrective (a crawl that
  # stands up) and a plain walk.
  @oracle ~w(crawl_to_stand walk_to_stand)
  # The operator's parity bound: 1% of the largest corrective offset on that clip.
  @parity_share 0.01

  def main do
    case run() do
      {:ok, out} -> IO.puts("BRIDGES-OK #{out}"); 0
      {:error, r} -> IO.puts(:stderr, "BRIDGES-FAIL #{if is_binary(r), do: r, else: inspect(r)}"); 1
    end
  end

  defp run do
    with {:ok, root} <- workspace_root(),
         cache = Path.join(root, ".bridges"),
         :ok <- File.mkdir_p(Path.join(cache, ".done")) do
      ctx = %{root: root, cache: cache, mmg: Path.expand("..", __DIR__),
              stamp: Calendar.strftime(DateTime.utc_now(), "%Y%m%d")}

      Enum.reduce_while(@steps, {:ok, cache}, fn step, acc ->
        marker = Path.join([cache, ".done", step])

        cond do
          File.exists?(marker) -> IO.puts("== #{step}: done in an earlier run"); {:cont, acc}
          true ->
            IO.puts("== #{step}")

            case apply(__MODULE__, String.to_atom("step_" <> step), [ctx]) do
              :ok -> File.write(marker, ctx.stamp); {:cont, acc}
              {:deferred, why} -> IO.puts("   deferred: #{why}"); {:cont, acc}
              {:error, r} -> {:halt, {:error, {String.to_atom(step), r}}}
            end
        end
      end)
    end
  end

  defp workspace_root do
    [System.get_env("INIT_CWD"), File.cwd!(), Path.expand(__DIR__)]
    |> Enum.reject(&is_nil/1)
    |> Enum.flat_map(&(Stream.iterate(&1, fn d -> Path.dirname(d) end) |> Enum.take(8)))
    |> Enum.find(&File.dir?(Path.join(&1, ".repo")))
    |> case do
      nil -> {:error, "no .repo above #{File.cwd!()}: run from the smoke-water client root"}
      root -> {:ok, root}
    end
  end

  # ---- steps ------------------------------------------------------------------------------------

  def step_tools(ctx) do
    src = Path.join(ctx.root, "3-interactor/motion-guest/tools/motion_native")
    b = Path.join(ctx.cache, "motion_native")

    with {:ok, _} <- cap("cmake", ["-S", src, "-B", b, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]),
         {:ok, _} <- cap("cmake", ["--build", b]),
         {:ok, _} <- cap("c++", ["-std=c++20", "-O2", "-ffp-contract=off", "-I", Path.join(ctx.mmg, "src"),
                                 Path.join(ctx.mmg, "tests/mm_kinds.cpp"), "-o", Path.join(ctx.cache, "mm_kinds")]) do
      need(ctx.cache, ["motion_native/motion_native", "mm_kinds"])
    end
  end

  def step_weights(ctx) do
    w = Path.join(ctx.cache, "w")

    with {:ok, tok} <- cap("bao-hf-token", []),
         env = [{"HF_TOKEN", String.trim(tok)}, {"PYTHONPATH", ""}],
         {:ok, _} <- cap("hf", ["download", @kimodo_repo, @kimodo_file, "--local-dir", Path.join(w, "Kimodo-SOMA-RP-v1.1-gguf")], env),
         {:ok, _} <- cap("hf", ["download", @text_repo, "MANIFEST.json", "--local-dir", Path.join(w, "text")], env),
         {:ok, _} <- cap("hf", ["download", @text_repo, "--include", @text_dir <> "/*", "--local-dir", Path.join(w, "text")], env),
         {:ok, man} <- File.read(Path.join([w, "text", "MANIFEST.json"])),
         {:ok, %{"files" => files}} <- decode(man) do
      Enum.reduce_while(files, :ok, fn f, _ ->
        want = f["sha256"]

        case sha256(Path.join([w, "text", f["path"]])) do
          {:ok, ^want} -> {:cont, :ok}
          {:ok, h} -> {:halt, {:error, "#{f["path"]} sha256 #{h}, the manifest says #{want}"}}
          e -> {:halt, e}
        end
      end)
    end
  end

  def step_engine(ctx) do
    src = Path.join(ctx.root, "5-repository/godot-double")
    name = "godot.macos.editor.double.arm64"
    out = Path.join(ctx.cache, name)

    with {:ok, gz} <- File.read(Path.join(src, name <> ".gz")),
         bin = :zlib.gunzip(gz),
         :ok <- File.write(out, bin),
         :ok <- File.chmod(out, 0o755),
         {:ok, sums} <- File.read(Path.join(src, "SHA256SUMS")),
         want when is_binary(want) <- sum_for(sums, name) || {:error, "no #{name} in SHA256SUMS"},
         {:ok, got} <- sha256(out) do
      if got == want, do: :ok, else: {:error, "#{name} sha256 #{got}, SHA256SUMS says #{want}"}
    end
  end

  def step_sandbox(ctx) do
    d = Path.join(ctx.cache, "sandbox")
    addon = Path.join(ctx.root, "4-entities/godot-sandbox/bin/addons/godot_sandbox")
    fw = "libgodot_riscv.macos.template_release.double.universal.framework"

    with {:ok, run, sha} <- green_run("V-Sekai-fire/godot-sandbox", "Builds"),
         {:ok, _} <- File.rm_rf(d),
         {:ok, _} <- cap("gh", ["run", "download", run, "-R", "V-Sekai-fire/godot-sandbox", "-n",
                                "godot-sandbox-macos-universal-double", "-D", d]),
         [lib | _] <- Path.wildcard(Path.join(d, "**/#{fw}")) ++ [{:error, "no #{fw} in the artifact"}],
         {:ok, _} <- File.rm_rf(Path.join(d, "godot_sandbox")),
         {:ok, _} <- File.cp_r(addon, Path.join(d, "godot_sandbox")),
         {:ok, _} <- File.cp_r(lib, Path.join([d, "godot_sandbox", "bin", fw])) do
      File.write(Path.join(d, "SOURCE"), "V-Sekai-fire/godot-sandbox Builds run #{run} (#{sha})\n")
    end
  end

  def step_guest(ctx) do
    d = Path.join(ctx.cache, "guest")
    r = "V-Sekai-fire/interactor-motion-guest"

    with {:ok, run, sha} <- green_run(r, "guest"),
         {:ok, json} <- cap("gh", ["api", "repos/#{r}/actions/runs/#{run}/artifacts"]),
         {:ok, %{"artifacts" => arts}} <- decode(json),
         name when is_binary(name) <- Enum.find_value(arts, &(String.starts_with?(&1["name"], "motion-elf-") && &1["name"])) ||
           {:error, "#{r} run #{run} has no motion-elf artifact"},
         {:ok, _} <- File.rm_rf(d),
         {:ok, _} <- cap("gh", ["run", "download", run, "-R", r, "-n", name, "-D", d]),
         :ok <- need(d, ["motion.elf"]) do
      File.write(Path.join(d, "SOURCE"), "#{r} guest run #{run} (#{sha}) artifact #{name}\n")
    end
  end

  def step_control(ctx) do
    key = "crouch_idle"
    db = Path.join([ctx.mmg, "data/kimodo-locomotion", key])
    c = Path.join(ctx.cache, "control")

    with {:ok, sk} <- File.read(Path.join(db, "skeleton.txt")),
         [_, prompt] <- Regex.run(~r/^prompt (.+)$/m, sk) || {:error, "#{key} has no prompt line"},
         :ok <- File.mkdir_p(c),
         :ok <- File.write(Path.join(c, "prompts.tsv"), "#{key}\t#{prompt}\n"),
         {:ok, _} <- embed(ctx, Path.join(c, "prompts.tsv"), Path.join(c, "emb")),
         {:ok, _} <- generate(ctx, Path.join(c, "prompts.tsv"), Path.join(c, "emb"), Path.join(c, "clips")),
         {:ok, log} <- cap(Path.join(ctx.cache, "motion_native/motion_native"),
                           ["compare", Path.join([c, "clips", key, "transforms.f32"]), Path.join(db, "transforms.f32"), "1e-3"]) do
      IO.puts(String.trim(log))
      :ok
    end
  end

  def step_embed(ctx), do: ok(embed(ctx, prompts(ctx), Path.join(ctx.cache, "emb")))

  def step_generate(ctx), do: ok(generate(ctx, prompts(ctx), Path.join(ctx.cache, "emb"), Path.join(ctx.cache, "clips")))

  def step_kinds(ctx) do
    kinds = Path.join(ctx.cache, "kinds")

    with :ok <- File.mkdir_p(kinds),
         {:ok, _} <- cap(Path.join(ctx.cache, "mm_kinds"), [kinds | db_dirs(ctx) ++ bridge_dirs(ctx)]) do
      need(kinds, Enum.map(bridge_keys(ctx), &(&1 <> ".kinds")))
    end
  end

  def step_assets(ctx) do
    d = Path.join(ctx.cache, "somax")

    with {:ok, _} <- File.rm_rf(d),
         {:ok, _} <- cap("gh", ["release", "download", @assets_tag, "-R", @assets_repo, "-D", d]),
         [tar] <- Path.wildcard(Path.join(d, "soma-assets-*.tar.zst")) ++ [{:error, "no soma-assets tarball in #{@assets_tag}"}],
         {:ok, _} <- cap("sh", ["-c", ~s(zstd -dc "$0" | tar -xf - -C "$1"), tar, d]),
         {:ok, sums} <- File.read(Path.join(d, "SHA256SUMS.txt")) do
      sums
      |> String.split("\n", trim: true)
      |> Enum.reduce_while(:ok, fn l, _ ->
        [want, f] = String.split(l, ~r/\s+/, parts: 2)

        case sha256(Path.join([d, "soma-assets", f])) do
          {:ok, ^want} -> {:cont, :ok}
          {:ok, got} -> {:halt, {:error, "#{f} sha256 #{got}, SHA256SUMS.txt says #{want}"}}
          e -> {:halt, e}
        end
      end)
    end
  end

  def step_kernels(ctx) do
    k = Path.join(ctx.root, "2-contract/anny-kernels")
    b = Path.join(ctx.cache, "anny-kernels")

    with {:ok, _} <- cap("bash", [Path.join(k, "kernels/anny/gen.sh"), "--no-emit"], [{"BUILD_DIR", b}]) do
      need(Path.join(b, "spv-anny"), Enum.map(@kernels, &(&1 <> ".spv")))
    end
  end

  def step_body(ctx) do
    b = Path.join(ctx.cache, "body")
    fx = Path.join(b, "fixture")
    out = Path.join(b, "gpu")

    with {:ok, _} <- File.rm_rf(b),
         {:ok, log} <- cap("python", [Path.join(ctx.mmg, "tools/bridges/body_frames.py"), Path.join(ctx.root, "2-contract/anny-kernels"),
                                      Path.join(ctx.cache, "somax/soma-assets"), fx, "--oracle=" <> Enum.join(@oracle, ",")
                                      | bridge_dirs(ctx) ++ db_dirs(ctx)], [{"PYTHONPATH", ""}, {"PYTHONHOME", nil}, {"VIRTUAL_ENV", nil}]),
         _ = log |> String.split("\n") |> Enum.filter(&String.starts_with?(&1, "ORACLE")) |> Enum.each(&IO.puts/1),
         {:ok, p} <- project(ctx),
         {:ok, glog} <- cap(Path.join(ctx.cache, "godot.macos.editor.double.arm64"),
                            ["--path", p, "--script", "res://body.gd", "--", "--fixture=" <> fx,
                             "--spv=" <> Path.join(ctx.cache, "anny-kernels/spv-anny"), "--out=" <> out,
                             "--bridges=" <> Enum.join(bridge_keys(ctx), ","), "--oracle=" <> Enum.join(@oracle, ","), "--joins"]),
         true <- String.contains?(glog, "RESULT: PASS") || {:error, "body.gd: " <> tail(glog)},
         _ = glog |> String.split("\n") |> Enum.filter(&String.starts_with?(&1, "BODY-GPU")) |> Enum.each(&IO.puts/1),
         {:ok, report} <- parity(fx, out) do
      File.write(Path.join(b, "parity.txt"), report)
    end
  end

  def step_joins(ctx) do
    out = Path.join(ctx.cache, "body/gpu")

    with {:ok, rows} <- joins(out, Path.join(ctx.cache, "kinds"), bridge_keys(ctx)) do
      head = "join\tkind\tdatabase frame\tmax_mm\tmean_mm\toff_frame\toff_max_mm\toff_mean_mm\tself_mm\n"
      tsv = Enum.map_join(rows, "", fn r -> Enum.join(r.cols, "\t") <> "\n" end)
      Enum.each(rows, &IO.puts(&1.line))
      bad = Enum.reject(rows, & &1.ok)

      with :ok <- File.write(Path.join(ctx.cache, "joins.tsv"), head <> tsv) do
        if bad == [], do: :ok, else: {:error, "join controls: " <> Enum.map_join(bad, "; ", & &1.line)}
      end
    end
  end

  def step_sheets(_ctx), do: {:deferred, "body orbit sheets wait for the operator (render_sheets/1)"}

  def step_video(_ctx), do: {:deferred, "CineForm MKVs wait for the operator (render_video/1)"}

  # Body orbit views (contract-orbit-views): a row per frame, 60 frames a sheet, the views from
  # sphere_hammersley_sequence; the zero-views control must fail; check_orbit_views.py checks every
  # bundle; a desk copy goes to ~/Desktop/lookdev-contact-sheets. Not run yet (deferred).
  def render_sheets(ctx) do
    clips = Path.join(ctx.cache, "clips")
    kinds = Path.join(ctx.cache, "kinds")
    sheets = Path.join(ctx.cache, "sheets")
    desk = Path.expand("~/Desktop/lookdev-contact-sheets")
    chart = Path.join(ctx.root, "2-contract/orbit-views/chart24.json")
    keys = bridge_keys(ctx)

    with :ok <- File.mkdir_p(sheets),
         :ok <- zero_views_fails(ctx, clips, hd(keys), kinds, Path.join(ctx.cache, "control-sheets")),
         :ok <- Enum.reduce_while(keys, :ok, fn k, _ ->
           case sheets_for(ctx, k, clips, kinds, sheets, chart) do
             :ok -> {:cont, :ok}
             e -> {:halt, e}
           end
         end),
         {:ok, log} <- cap("uv", ["run", Path.join(ctx.root, "2-contract/orbit-views/check_orbit_views.py"), sheets]),
         :ok <- File.mkdir_p(desk) do
      IO.puts(String.trim(log))
      Path.wildcard(Path.join(sheets, "*.preview.png"))
      |> Enum.each(&File.cp(&1, Path.join(desk, "walk-bridges-" <> Path.basename(&1))))
    end
  end

  # One CineForm MKV per clip (the four orbit views live, one clip frame per movie frame) from the
  # godot-cineform engine; ffprobe must read F frames and every frame must differ from the others
  # (the status line carries the frame number), or the window skipped draws. Not run yet (deferred).
  def render_video(ctx) do
    eng = Path.join(ctx.root, "4-entities/godot-cineform/bin/godot.macos.editor.double.arm64")
    dir = Path.join(ctx.cache, "video")

    with :ok <- File.mkdir_p(dir),
         {:ok, p} <- project(ctx) do
      Enum.reduce_while(bridge_keys(ctx), :ok, fn k, _ ->
        clip = Path.join([ctx.cache, "clips", k])

        with {:ok, sk} <- File.read(Path.join(clip, "skeleton.txt")),
             [_, n, fps] <- Regex.run(~r/^frames (\d+) fps ([\d.]+)/m, sk) || {:error, "#{k} has no frames line"},
             stem = next_stem(dir, "#{ctx.stamp}_motion-matching_#{String.replace(k, "_", "-")}-body", ".mkv"),
             mkv = Path.join(dir, stem <> ".mkv"),
             {:ok, _} <- cap(eng, ["--path", p, "--fixed-fps", "#{round(String.to_float(fps))}", "--write-movie", mkv,
                                   "--script", "res://orbit.gd", "--", "--clip=" <> clip, "--body=" <> Path.join(ctx.cache, "body/gpu"),
                                   "--fixture=" <> Path.join(ctx.cache, "body/fixture"), "--kinds=" <> Path.join([ctx.cache, "kinds", k <> ".kinds"]),
                                   "--from=0", "--to=#{String.to_integer(n) - 1}", "--movie"]),
             {:ok, probe} <- cap("ffprobe", ["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                             "stream=codec_name,nb_read_frames", "-of", "csv=p=0", mkv]),
             {:ok, md5} <- cap("ffmpeg", ["-v", "error", "-i", mkv, "-map", "0:v", "-f", "framemd5", "-"]),
             uniq = md5 |> String.split("\n") |> Enum.reject(&String.starts_with?(&1, "#")) |> Enum.map(&List.last(String.split(&1, ","))) |> Enum.uniq() |> length(),
             true <- (String.trim(probe) == "cfhd,#{n}" and uniq == String.to_integer(n)) ||
                       {:error, "#{stem}: ffprobe #{String.trim(probe)}, #{uniq} distinct frames, want #{n}"} do
          IO.puts("video #{stem} #{String.trim(probe)}")
          File.cp(mkv, Path.join(Path.expand("~/Desktop/lookdev-contact-sheets"), "walk-bridges-" <> stem <> ".mkv"))
          {:cont, :ok}
        else
          e -> {:halt, e}
        end
      end)
    end
  end

  # ---- helpers ----------------------------------------------------------------------------------

  defp prompts(ctx), do: Path.join(ctx.mmg, "tools/bridges/prompts.tsv")

  defp bridge_keys(ctx), do: Path.join(ctx.cache, "clips") |> File.ls!() |> Enum.sort()

  defp bridge_dirs(ctx), do: Enum.map(bridge_keys(ctx), &Path.join([ctx.cache, "clips", &1]))

  defp db_dirs(ctx), do: Path.wildcard(Path.join(ctx.mmg, "data/*/*")) |> Enum.filter(&File.dir?/1)

  # GPU vertices against SOMA-X's: the half-angle form (SOMA-X's own) must be within the bound, the
  # correctives-off control must not; the default Kabsch form's distance is reported beside them.
  defp parity(fx, out) do
    with {:ok, m} <- File.read(Path.join(fx, "meta.json")),
         {:ok, %{"oracle" => orc}} <- decode(m) do
      Enum.reduce_while(@oracle, {:ok, ""}, fn k, {:ok, acc} ->
        bound = orc[k]["corrective_mm"] / 1000 * @parity_share

        with {:ok, ref} <- File.read(Path.join([fx, "oracle", k <> ".on.f32"])),
             {:ok, s} <- vdist(Path.join(out, k <> ".soma.f32"), ref),
             {:ok, o} <- vdist(Path.join(out, k <> ".off.f32"), ref),
             {:ok, kb} <- vdist(Path.join(out, k <> ".verts.f32"), ref) do
          line = :io_lib.format("PARITY ~s bound ~.4f mm (1% of corrective ~.2f mm): gpu ~.4f max ~.5f mean mm ~s; " <>
                                  "correctives-off control ~.3f max mm ~s; Kabsch default form ~.3f max ~.4f mean mm (info)~n",
                                [k, bound * 1000, orc[k]["corrective_mm"], elem(s, 0) * 1000, elem(s, 1) * 1000,
                                 if(elem(s, 0) <= bound, do: "PASS", else: "FAIL"), elem(o, 0) * 1000,
                                 if(elem(o, 0) > bound, do: "FAILS as it must", else: "PASSED (broken)"),
                                 elem(kb, 0) * 1000, elem(kb, 1) * 1000]) |> IO.iodata_to_binary()
          IO.write(line)

          if elem(s, 0) <= bound and elem(o, 0) > bound,
            do: {:cont, {:ok, acc <> line}},
            else: {:halt, {:error, String.trim(line)}}
        else
          e -> {:halt, e}
        end
      end)
    else
      {:ok, _} -> {:error, "meta.json has no oracle clips"}
      e -> e
    end
  end

  defp vdist(path, ref) do
    with {:ok, a} <- File.read(path) do
      if byte_size(a) == byte_size(ref), do: {:ok, vd(a, ref, 0.0, 0.0, 0)}, else: {:error, "#{path}: size differs from the oracle"}
    end
  end

  defp vd(<<ax::float-32-little, ay::float-32-little, az::float-32-little, ra::binary>>,
          <<bx::float-32-little, by::float-32-little, bz::float-32-little, rb::binary>>, mx, s, n) do
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    d = :math.sqrt(dx * dx + dy * dy + dz * dz)
    vd(ra, rb, max(mx, d), s + d, n + 1)
  end

  defp vd(<<>>, <<>>, mx, s, n), do: {mx, s / max(n, 1)}

  defp floats(path) do
    with {:ok, b} <- File.read(path), do: {:ok, for(<<x::float-32-little <- b>>, do: x) |> List.to_tuple()}
  end

  defp kinds_of(path) do
    with {:ok, t} <- File.read(path),
         do: {:ok, t |> String.split("\n", trim: true) |> Enum.map(&String.to_integer/1) |> List.to_tuple()}
  end

  # body.gd's joins/<clip>.f32: frames x targets x (max, mean) metres. For each target (a bridge's
  # first or last frame) the join is the closest database frame of the same kind by mean distance
  # (all database frames when none has that kind); controls: the same join one frame off is larger,
  # and the target against its own frame is 0 (the batched GPU path kept frame order).
  defp joins(out, kinds, bridges) do
    with {:ok, tn} <- File.read(Path.join([out, "joins", "targets.txt"])) do
      targets = String.split(tn, "\n", trim: true)
      ntg = length(targets)
      clips = Path.wildcard(Path.join([out, "joins", "*.f32"])) |> Enum.map(&Path.basename(&1, ".f32"))
      data = Map.new(clips, fn c -> {c, elem(floats(Path.join([out, "joins", c <> ".f32"])), 1)} end)
      kd = Map.new(clips, fn c -> {c, kinds_of(Path.join(kinds, c <> ".kinds"))} end)

      case Enum.find(clips, &(not match?({:ok, _}, kd[&1]))) do
        nil ->
          kd = Map.new(kd, fn {c, {:ok, k}} -> {c, k} end)
          nf = fn c -> div(tuple_size(data[c]), ntg * 2) end
          at = fn c, f, t -> {elem(data[c], (f * ntg + t) * 2), elem(data[c], (f * ntg + t) * 2 + 1)} end
          db = clips -- bridges
          mm = &:erlang.float_to_binary(&1 * 1000, decimals: 1)

          rows =
            targets
            |> Enum.with_index()
            |> Enum.map(fn {name, t} ->
              [b, fs] = String.split(name, "@")
              fb = String.to_integer(fs)
              kt = elem(kd[b], fb)
              same = for c <- db, f <- 0..(nf.(c) - 1), elem(kd[c], f) == kt, do: {c, f}
              cands = if same == [], do: for(c <- db, f <- 0..(nf.(c) - 1), do: {c, f}), else: same
              {c, f} = Enum.min_by(cands, fn {c, f} -> elem(at.(c, f, t), 1) end)
              {mx, mean} = at.(c, f, t)
              off = if f + 1 < nf.(c), do: f + 1, else: f - 1
              {omx, omean} = at.(c, off, t)
              {smx, _} = at.(b, fb, t)
              ok = omean > mean and smx == 0.0
              side = if fb == 0, do: "into", else: "out of"

              %{ok: ok,
                cols: [name, kt, "#{c}@#{f}", mm.(mx), mm.(mean), "#{c}@#{off}", mm.(omx), mm.(omean), mm.(smx)],
                line: "JOIN #{side} #{b} (#{name}, kind #{kt}) <-> #{c}@#{f}: max #{mm.(mx)} mean #{mm.(mean)} mm; " <>
                      "one frame off #{c}@#{off}: max #{mm.(omx)} mean #{mm.(omean)} mm #{if omean > mean, do: "larger as it must", else: "NOT larger"}; " <>
                      "self #{mm.(smx)} mm"}
            end)

          {:ok, rows}

        c ->
          {:error, "no kinds for #{c}"}
      end
    end
  end

  defp embed(ctx, tsv, out) do
    with {:ok, text} <- File.read(tsv) do
      pairs = text |> String.split("\n", trim: true) |> Enum.map(&String.replace(&1, "\t", "=", global: false))
      cap(Path.join(ctx.cache, "motion_native/motion_native"),
          ["embed", Path.join([ctx.cache, "w/text", @text_dir]), out | pairs])
    end
  end

  defp project(ctx) do
    p = Path.join(ctx.cache, "project")

    with {:ok, _} <- File.rm_rf(p),
         {:ok, _} <- File.cp_r(Path.join(ctx.mmg, "tools/bridges"), p),
         :ok <- File.mkdir_p(Path.join(p, "addons")),
         {:ok, _} <- File.cp_r(Path.join(ctx.cache, "sandbox/godot_sandbox"), Path.join(p, "addons/godot_sandbox")),
         :ok <- File.mkdir_p(Path.join(p, ".godot")),
         :ok <- File.write(Path.join(p, ".godot/extension_list.cfg"), "res://addons/godot_sandbox/bin/godot-riscv.gdextension\n") do
      {:ok, p}
    end
  end

  # Windowed on the GPU: motion.elf needs a local RenderingDevice.
  defp generate(ctx, tsv, emb, out) do
    with {:ok, p} <- project(ctx),
         {:ok, log} <- cap(Path.join(ctx.cache, "godot.macos.editor.double.arm64"),
                           ["--path", p, "--", "--elf=" <> Path.join(ctx.cache, "guest/motion.elf"),
                            "--weights=" <> Path.join(ctx.cache, "w"), "--emb=" <> emb, "--prompts=" <> tsv,
                            "--out=" <> out, "--seconds=#{@seconds}", "--seed=#{@seed}"]) do
      oks = Regex.scan(~r/^GEN-OK .*$/m, log) |> List.flatten()
      Enum.each(oks, &IO.puts/1)
      if String.contains?(log, "GEN-DONE bad=0"), do: {:ok, oks}, else: {:error, "generation failed: " <> tail(log)}
    end
  end

  defp orbit(ctx, args) do
    with {:ok, p} <- project(ctx) do
      cap(Path.join(ctx.cache, "godot.macos.editor.double.arm64"), ["--path", p, "--script", "res://orbit.gd", "--" | args])
    end
  end

  defp zero_views_fails(ctx, clips, key, kinds, sheets) do
    args = ["--clip=" <> Path.join(clips, key), "--kinds=" <> Path.join(kinds, key <> ".kinds"), "--body=" <> Path.join(ctx.cache, "body/gpu"),
            "--fixture=" <> Path.join(ctx.cache, "body/fixture"), "--out=" <> Path.join(sheets, "zero.exr"), "--from=0", "--to=0",
            "--control=zero_views"]

    case orbit(ctx, args) do
      {:error, _} -> IO.puts("zero-views control: FAIL as it must"); :ok
      {:ok, _} -> {:error, "the zero-views control passed; the blank check is broken"}
    end
  end

  defp sheets_for(ctx, key, clips, kinds, sheets, chart) do
    with {:ok, sk} <- File.read(Path.join([clips, key, "skeleton.txt"])),
         [_, n] <- Regex.run(~r/^frames (\d+)/m, sk) || {:error, "#{key} has no frames line"} do
      frames = String.to_integer(n)

      0..(frames - 1)//@rows_per_sheet
      |> Enum.reduce_while(:ok, fn from, _ ->
        to = min(from + @rows_per_sheet - 1, frames - 1)
        stem = next_stem(sheets, "#{ctx.stamp}_motion-matching_#{String.replace(key, "_", "-")}-f#{pad(from)}-#{pad(to)}")
        out = Path.join(sheets, stem <> ".exr")
        args = ["--clip=" <> Path.join(clips, key), "--kinds=" <> Path.join(kinds, key <> ".kinds"),
                "--body=" <> Path.join(ctx.cache, "body/gpu"), "--fixture=" <> Path.join(ctx.cache, "body/fixture"), "--out=" <> out, "--from=#{from}", "--to=#{to}", "--chart=" <> chart]

        with {:ok, log} <- orbit(ctx, args),
             true <- String.contains?(log, "RESULT: PASS") || {:error, "#{stem}: " <> tail(log)},
             :ok <- cff(ctx, Path.join(sheets, stem <> ".cff"), key, from, to) do
          IO.puts("sheet #{stem}")
          {:cont, :ok}
        else
          e -> {:halt, e}
        end
      end)
    end
  end

  defp next_stem(dir, base, ext \\ ".exr") do
    Stream.iterate(1, &(&1 + 1))
    |> Enum.find(fn i -> not File.exists?(Path.join(dir, "#{base}_#{String.pad_leading("#{i}", 4, "0")}#{ext}")) end)
    |> then(&"#{base}_#{String.pad_leading("#{&1}", 4, "0")}")
  end

  defp pad(i), do: String.pad_leading("#{i}", 3, "0")

  defp cff(ctx, path, key, from, to) do
    sha =
      case cap("git", ["-C", ctx.mmg, "rev-parse", "HEAD"]) do
        {:ok, s} -> String.trim(s)
        _ -> "unknown"
      end

    File.write(path, """
    cff-version: 1.2.0
    title: "Walk-mode bridge clip #{key}, frames #{from}-#{to}, orbit views"
    message: "Kimodo-SOMA-RP-v1.1 output on motion.elf (NVIDIA Open Model License); posed on the SOMA-2603.16858 body on the GPU (tools/bridges/body.gd), drawn by tools/bridges/orbit.gd."
    version: "untagged"
    date-released: "#{Date.utc_today()}"
    commit: "#{sha}"
    license: MIT
    url: "https://github.com/v-sekai-fire"
    authors:
      - family-names: Lee
        given-names: "K. S. Ernest (iFire)"
    """)
  end

  defp green_run(repo, workflow) do
    with {:ok, json} <- cap("gh", ["run", "list", "-R", repo, "-w", workflow, "-b", "main", "--status", "success",
                                   "-L", "1", "--json", "databaseId,headSha"]),
         {:ok, [%{"databaseId" => id, "headSha" => sha}]} <- decode(json) do
      {:ok, "#{id}", sha}
    else
      {:ok, _} -> {:error, "#{repo} has no green #{workflow} run on main"}
      e -> e
    end
  end

  defp sum_for(sums, name) do
    Enum.find_value(String.split(sums, "\n", trim: true), fn l ->
      case String.split(l, ~r/\s+/, parts: 2) do
        [h, ^name] -> h
        _ -> nil
      end
    end)
  end

  defp sha256(path) do
    case File.open(path, [:read, :binary]) do
      {:ok, f} ->
        h = IO.binstream(f, 1 <<< 24) |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
        File.close(f)
        {:ok, :crypto.hash_final(h) |> Base.encode16(case: :lower)}

      e ->
        e
    end
  end

  defp need(dir, files) do
    case Enum.reject(files, &File.exists?(Path.join(dir, &1))) do
      [] -> :ok
      missing -> {:error, "missing #{Enum.join(missing, ", ")} in #{dir}"}
    end
  end

  defp decode(json) do
    case :json.decode(json) do
      v -> {:ok, v}
    end
  rescue
    e -> {:error, "bad JSON: #{Exception.message(e)}"}
  end

  defp ok({:ok, _}), do: :ok
  defp ok(e), do: e

  defp tail(s), do: s |> String.split("\n") |> Enum.take(-8) |> Enum.join("\n")

  defp cap(cmd, args, env \\ []) do
    case System.find_executable(cmd) || (File.exists?(cmd) && cmd) do
      false -> {:error, "#{cmd} is not on PATH"}
      exe ->
        case System.cmd(exe, args, env: env, stderr_to_stdout: true) do
          {out, 0} -> {:ok, out}
          {out, code} -> {:error, "#{Path.basename(cmd)} exited #{code}: #{tail(out)}"}
        end
    end
  end
end

System.halt(Bridges.main())
