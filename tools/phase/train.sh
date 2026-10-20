export PATH=$HOME/.pixi/bin:$PATH MPLBACKEND=Agg
G=${G:-/root/kimodo/wt/mmg-phase60}; W=${W:-/root/kimodo/wt/wtd-mm}
git -C /root/kimodo/repos/motion-matching-guest.git fetch -q origin && git -C $G merge -q --ff-only origin/dev/phase-60fps && git -C $G log --oneline -1 || { echo "STALE GUEST CHECKOUT"; exit 1; }
grep -q "WTD-SKIP" $G/tools/phase/prep_wtd.py || { echo "PREP STILL RESAMPLES"; exit 1; }
cd $W && rm -rf Datasets/mm results/mm-o3de60 && mkdir -p logs
pixi run python $G/tools/phase/prep_wtd.py $G/data/kimodo-locomotion:$G/data/o3de-motionmatching Datasets/mm --fps 60 | grep -v WTD-SKIP
sed -n 6p Datasets/mm/Description.txt | grep -qx 60 || { echo "DATASET IS NOT 60 FPS"; exit 1; }
CUDA_VISIBLE_DEVICES=0 pixi run python train_vq.py --load=mm --save=./results/mm-o3de60 --use_tqdm=0 --std_cap=0.001 --fps=60 --window=2.0 --epochs=50 --num_embed_vq=8 --restart_period=50 --restart_mult=1 --plotting_interval=100000000 --log_every=50 > logs/train-o3de60.log 2>&1 < /dev/null
echo "TRAIN-EXIT $?"
tr '\r' '\n' < logs/train-o3de60.log | grep -iE "Traceback|Error" | tail -3
CUDA_VISIBLE_DEVICES=0 pixi run python test_vq.py --save=./results/mm-o3de60 --plot_cnt=0 > logs/test-o3de60.log 2>&1 < /dev/null
echo "TEST-EXIT $?"
pixi run python $G/tools/phase/write_phase.py results/mm-o3de60/Manifolds_0_final.npz Datasets/mm --shift | tail -1
pixi run python $G/tools/phase/write_phase.py results/mm-o3de60/Manifolds_0_final.npz Datasets/mm > logs/gate-o3de60.txt; echo "GATE-EXIT $?"
cat logs/gate-o3de60.txt
